from collections import defaultdict
from functools import cached_property
import yaml
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Literal, Union

import torch
import typer
import xxhash
from datasets import Dataset, fingerprint
from tqdm.auto import tqdm
from pydantic import BaseModel
from rich import print
from transformers import AutoTokenizer, AutoModelForCausalLM
import jinja2


"""
Policy annotation toolkit.

Score a bunch of raw text against a user-configurable set of policies.
A LlamaGuard2-based model will score each line of text against each policy.

Data workflow:
- Each line of the input text becomes a sample in the dataset, including
  context lines above (like grep -A 3)
- For each policy, we create a prompt template by combining the policy
  details/examples with a model-specific template.
- Input is passed through model.forward once to get logits for each token.
"""

def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

class HashableModel(BaseModel):
    @property
    def hash(self):
        return xxhash.xxh64_hexdigest(self.model_dump_json())

class BaseAnnotator(HashableModel):
    """
    Class that knows how to annotate a dataset.
    """
    annotator_type: str # discriminated union tag

    def preprocess(self, dataset: Dataset) -> Dataset:
        """
        Apply preprocessing to the dataset before annotation:
        add model-specific context and assign a UID to each sample
        to aid caching.

        The dataset needs to have the following fields:
        - raw_text: the text to annotate

        After preprocessing, the dataset should have at least:
        - uid: a unique identifier for each sample
        """
        ...
    def annotate(self, dataset: Dataset) -> Generator["AnnotationResult"]:
        ...

class LlamaGuard2Policy(HashableModel):
    """
    A single policy to score separately, like 'hate speech',
    'sexual content', ...
    """
    name: str
    details: str

class AnnotationResult(BaseModel):
    """One score of a single line of text against a single policy."""
    uid: str
    file_name: str
    file_line: int
    policy_name: str
    raw_text: str
    next_token: int
    score: float

class Llamaguard2Annotator(BaseAnnotator):
    """
    Annotator that uses LlamaGuard2-based models for text classification.

    Llamaguard-2 models take a configurable policy / list of categories
    and returns a static "safe" or "unsafe" scores for each category.

    To turn this into a threashold, we take the raw outputs for the "safe"
    and "unsafe" tokens at the end of the sequence. The model returns
    unnormalized log-probabilities for each token (incorrectly called
    "logits" by convention).

    We can then turn this into an actual probability by applying:
       sigmoid(unsafe - safe),
    which is equivalent to
       exp(unsafe) / (exp(unsafe) + exp(safe))
    but more numerically stable.

    It's generally better to use llamaguard-2 than llamaguard-4, because
    it was produced in April 2024, before Meta's policy change
    deprioritized hate speech. See
    https://krnel.ai/blog/2025-06-09-guardrail-comparison/ for a comparison.
    """
    annotator_type: Literal["llamaguard2"]
    model_id: str
    dtype: str = "bfloat16"
    batch_size: int = 1
    tokenizer_config: dict
    input_context_lines: int = 3
    template: str
    policies: list[LlamaGuard2Policy]

    def preprocess(self, dataset: Dataset) -> Dataset:
        """
        Add context lines to the dataset and expand the prompt template
        for all policies.
        """
        def _add_context_lines(batch):
            batch["context"] = [
                "\n".join(batch["raw_text"][max(0, i-self.input_context_lines):i+1])
                for i in range(0, len(batch["raw_text"]))
            ]
            return batch
        dataset = dataset.map(
            _add_context_lines, batched=True, batch_size=len(dataset)
        )

        policy_template = jinja2.Template(self.template)
        def _expand_policies(_batch):
            # Expand the prompt template for all policies
            batch = defaultdict(list)
            for k, v in _batch.items():
                batch[k] = v * len(self.policies)
            for policy in self.policies:
                # Render the template with the policy details
                rendered_text = policy_template.render(
                    policy_details=policy.details,
                    raw_text=batch["context"][0],
                )
                batch["input"].append(rendered_text)
                batch["policy_name"].append(policy.name)
                batch["uid"].append(
                    self.hash + xxhash.xxh64_hexdigest(rendered_text)
                )
            return batch
        return dataset.map(_expand_policies, batched=True, batch_size=1)

    def annotate(self, dataset: Dataset) -> Generator[AnnotationResult]:
        with self._load_model() as (tokenizer, model):
            if getattr(tokenizer, "pad_token", None) is None:
                tokenizer.pad_token = tokenizer.eos_token
            for batch in tqdm(dataset.batch(self.batch_size)):
                inputs = tokenizer(
                    batch["input"],
                    **self.tokenizer_config,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    padding_side='left',
                )
                inputs = {k: v.to(_get_device()) for k, v in inputs.items()}
                with torch.inference_mode():
                    outputs = model(**inputs)
                logits = outputs.logits
                batch['token_len'] = inputs['attention_mask'].sum(dim=-1)
                safe = logits[..., -1, tokenizer.vocab["▁safe"]]
                unsafe = logits[..., -1, tokenizer.vocab["▁unsafe"]]
                # these are logits, so apply our own softmax
                scores = torch.sigmoid(unsafe - safe)
                batch['score'] = scores
                batch['next_token'] = logits[..., -1, :].argmax(dim=-1)
                print("Batch:", batch)
                print("Logit shape:", logits.shape)
                for i in range(len(batch['input'])):
                    yield AnnotationResult.model_validate({
                        k: batch[k][i] for k in batch.keys()
                    })

    @contextmanager
    def _load_model(self) -> Generator[tuple[AutoTokenizer, AutoModelForCausalLM], None, None]:
        print(f"Loading LlamaGuard2 model [blue]{self.model_id}[/blue] "
          f"with dtype [blue]{self.dtype}[/blue] "
          f"on device [blue]{_get_device()}[/blue]")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map=_get_device(),
        )
        model.compile(backend='eager')

        with torch.inference_mode():
            yield tokenizer, model


class Config(BaseModel):
    database_path: Path
    annotations: list[Union[Llamaguard2Annotator]]


def main(
    input_file: Path,
    config_path: Path = Path("config.yaml"),
):
    config_data = yaml.safe_load(config_path.read_text())
    config_path = Config.model_validate(config_data)

    ds = Dataset.from_list(
        [{
            "raw_text": line,
            "file_name": str(input_file),
            "file_line": i + 1,
          } for i, line in enumerate(input_file.read_text().splitlines()) if line.strip()]
    )

    # TODO: strip out time codes in input
    # TODO: caching, for easier resumption

    print("Dataset:", ds)
    for annotator in config_path.annotations:
        preprocessed_ds = annotator.preprocess(ds)
        print(preprocessed_ds)
        for result in annotator.annotate(preprocessed_ds):
            print(result)


if __name__ == "__main__":
    typer.run(main)
