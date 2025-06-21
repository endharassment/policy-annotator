import yaml
from pathlib import Path
from typing import Literal, Union

import torch
import typer
from datasets import Dataset
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
    dtype: str = "float16"
    batch_size: int = 1
    tokenizer_config: dict
    input_context_lines: int = 3
    device: str = "cpu"
    template: str
    policies: list[LlamaGuard2Policy]

    def __call__(self, dataset: Dataset) -> Dataset:
        print(f"Loading LlamaGuard2 model [blue]{self.model_id}[/blue] with dtype [blue]{self.dtype}[/blue] on device [blue]{self.device}[/blue]")
        tokenizer, model = self._load_model()
        dataset.set_format(type="torch")
        dataset = dataset.map(
            lambda x: tokenizer(
                x["text"],
                **self.tokenizer_config,
                return_tensors="pt",
            ),
            batched=True,
            batch_size=self.batch_size,
        )

    def _load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map=self.device,
        )
        torch.set_grad_enabled(False)
        return tokenizer, model


class Config(BaseModel):
    database_path: Path
    annotations: list[Union[LlamaGuard2Annotation]]


def main(
    input_file: Path,
    config: Path = Path("config.yaml"),
):
    config_data = yaml.safe_load(config.read_text())
    annotation = Config.model_validate(config_data)

    ds = Dataset.from_list(
        [{"text": "This is a test sentence."}, {"text": "Another test sentence."}]
    )
    print(ds)
    for annotator in annotation.annotations:
        print(annotator(ds))


if __name__ == "__main__":
    typer.run(main)
