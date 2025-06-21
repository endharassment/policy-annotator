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

class HashableModel(BaseModel):
    @property
    def hash(self):
        return xxhash.xxh64_hexdigest(self.model_dump_json())

class BaseAnnotation(HashableModel):
    """
    Class that knows how to annotate a dataset.
    """
    annotator_type: str # discriminated union type
    def preprocess(self, dataset: Dataset) -> Dataset:
        ...
    def annotate(self, dataset: Dataset) -> Generator["AnnotationResult"]:
        ...

class LlamaGuard2Policy(HashableModel):
    """
    A single policy, like 'hate speech', 'sexual content', ...
    """
    name: str
    details: str

class AnnotationResult(BaseModel):
    uid: str
    file_line: int
    policy_name: str
    raw_text: str
    next_token: int
    score: float

class LlamaGuard2Annotation(BaseAnnotation):
    """
    Annotation that uses LlamaGuard2-based models for text classification.
    """
    annotator_type: Literal["llamaguard2"]
    model_id: str
    dtype: str = "bfloat16"
    batch_size: int = 1
    tokenizer_config: dict
    input_context_lines: int = 3
    device: str = "cpu"
    template: str
    policies: list[LlamaGuard2Policy]

    def preprocess(self, dataset: Dataset) -> Dataset:
        """
        Add context lines to the dataset and expand the prompt template for all policies.
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
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
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
                print(batch)
                print(logits.shape)
                for i in range(len(batch['input'])):
                    yield AnnotationResult.model_validate({
                        k: batch[k][i] for k in batch.keys()
                    })

    @contextmanager
    def _load_model(self) -> Generator[tuple[AutoTokenizer, AutoModelForCausalLM], None, None]:
        print(f"Loading LlamaGuard2 model [blue]{self.model_id}[/blue] with dtype [blue]{self.dtype}[/blue] on device [blue]{self.device}[/blue]")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map=self.device,
        )
        model.compile(backend='eager')

        with torch.inference_mode():
            yield tokenizer, model


class Config(BaseModel):
    database_path: Path
    annotations: list[Union[LlamaGuard2Annotation]]


def main(
    input_file: Path,
    config_path: Path = Path("config.yaml"),
):
    config_data = yaml.safe_load(config_path.read_text())
    config_path = Config.model_validate(config_data)

    ds = Dataset.from_list(
        [{
            "raw_text": line,
            "filename": str(input_file),
            "file_line": i + 1,
            "file_hash": xxhash.xxh64_hexdigest(line.encode("utf-8")),
          } for i, line in enumerate(input_file.read_text().splitlines()) if line.strip()]
    )

    print(ds)
    for annotator in config_path.annotations:
        preprocessed_ds = annotator.preprocess(ds)
        print(preprocessed_ds)
        print(preprocessed_ds[0])
        for result in annotator.annotate(preprocessed_ds):
            print(result)


if __name__ == "__main__":
    typer.run(main)
