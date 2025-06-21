from functools import cached_property
import yaml
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Literal, Union

import torch
import typer
import xxhash
from datasets import Dataset, fingerprint
from pydantic import BaseModel
from rich import print
from transformers import AutoTokenizer, AutoModelForCausalLM
import jinja2

class BaseAnnotation(BaseModel):
    """
    Class that knows how to annotate a dataset.
    """
    annotator_type: str # discriminated union type
    def __call__(self, dataset: Dataset) -> Dataset:
        ...

class LlamaGuard2Policy(BaseModel):
    """
    A single policy, like 'hate speech', 'sexual content', ...
    """
    name: str
    details: str

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
    scoring_function: str
    policies: list[LlamaGuard2Policy]

    def __call__(self, dataset: Dataset) -> Dataset:
        fingerprint.disable_caching()
        with self._load_model() as (tokenizer, model):
            if getattr(tokenizer, "pad_token", None) is None:
                print("Warning: No pad token found")
                tokenizer.pad_token = tokenizer.eos_token
            dataset.set_format(type="torch")

            # Add context lines to the raw text
            def _add_context_lines(batch):
                # Add context lines to the raw text
                batch["context"] = [
                    "\n".join(batch["raw_text"][max(0, i-self.input_context_lines):i+1])
                    for i in range(0, len(batch["raw_text"]))
                ]
                return batch
            dataset = dataset.map(_add_context_lines, batched=True, batch_size=len(dataset))

            policy_template = jinja2.Template(self.template)
            def _expand_policies(batch):
                # Expand the prompt template for all policies
                batch = {k: v*len(self.policies) for k, v in batch.items()}
                batch["input"] = []
                batch["policy_name"] = []
                for policy in self.policies:
                    # Render the template with the policy details
                    rendered_text = policy_template.render(
                        policy_details=policy.details,
                        raw_text=batch["context"][0],
                    )
                    batch["input"].append(rendered_text)
                    batch["policy_name"].append(policy.name)
                return batch
            dataset = dataset.map(_expand_policies, batched=True, batch_size=1)
            print(dataset)
            print(dataset[0])
            def _infer(batch):
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
                scores = logits[..., 0, tokenizer.vocab["▁unsafe"]] / (
                    logits[..., 0, tokenizer.vocab["▁safe"]]
                    + logits[..., 0, tokenizer.vocab["▁unsafe"]]
                )
                batch['score'] = scores
                print(batch)
                print(logits.shape)
                return batch
            dataset = dataset.map(
                _infer,
                batched=True,
                batch_size=self.batch_size,
            )
        return dataset

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
    config: Path = Path("config.yaml"),
):
    config_data = yaml.safe_load(config.read_text())
    config = Config.model_validate(config_data)

    ds = Dataset.from_list(
        [{
            "raw_text": line,
            "filename": input_file.name,
            "file_hash": xxhash.xxh64_hexdigest(line.encode("utf-8")),
          } for line in input_file.read_text().splitlines()]
    )

    print(ds)
    for annotator in config.annotations:
        ds = annotator(ds)


if __name__ == "__main__":
    typer.run(main)
