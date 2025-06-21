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
    policies: list[LlamaGuard2Policy]

    @cached_property
    def hash(self) -> str:
        return xxhash.xxh64_hexdigest(self.model_dump_json())

    def __call__(self, dataset: Dataset) -> Dataset:
        with self._load_model() as (tokenizer, model):
            if getattr(tokenizer, "pad_token", None) is None:
                print("Warning: No pad token found")
                tokenizer.pad_token = tokenizer.eos_token
            dataset.set_format(type="torch")
            print(dataset[0])
            def _infer(batch):
                inputs = tokenizer(
                    batch["raw_text"],
                    **self.tokenizer_config,
                    return_tensors="pt",
                )
                print(inputs['attention_mask'].sum(dim=-1))
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.inference_mode():
                    outputs = model(**inputs)
                logits = outputs.logits
                predictions = logits.argmax(dim=-1).cpu().numpy()
                return {"predictions": predictions}
            dataset = dataset.map(
                _infer,
                batched=True,
                batch_size=self.batch_size,
                new_fingerprint=self.hash,
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
        import IPython; IPython.embed()


if __name__ == "__main__":
    typer.run(main)
