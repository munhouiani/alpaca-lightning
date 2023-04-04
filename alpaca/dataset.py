import copy
from functools import partial
from typing import Sequence, Dict, Optional

import torch
from datasets import load_dataset, load_from_disk
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer


class AlpacaLightningDataModule(LightningDataModule):
    PROMPT_INPUT = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    )

    PROMPT_NO_INPUT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )

    IGNORE_INDEX = -100

    def __init__(self, tokenizer: PreTrainedTokenizer, batch_size: int, data_path: str):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.dataset = None

    def prepare_data(self) -> None:
        # load data
        dataset = load_dataset("json", data_files=self.data_path)
        # add prompt
        dataset = dataset.map(
            self.add_prompt_function,
            batched=True,
            remove_columns=["input", "instruction"],
        )
        # add eos
        dataset = dataset.map(
            self.add_eos_token_function, batched=True, remove_columns=["output"]
        )
        # tokenize
        dataset = dataset.map(
            self.tokenize_function, batched=True, remove_columns=["source", "target"]
        )
        # save to disk
        dataset.save_to_disk("alpaca_dataset")

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = load_from_disk("alpaca_dataset").with_format("torch")

    def add_prompt_function(self, examples):
        source = []
        for _input, _instruction in zip(examples["input"], examples["instruction"]):
            if _input != "":
                prompt = self.PROMPT_INPUT
            else:
                prompt = self.PROMPT_NO_INPUT

            result = prompt.format_map({"input": _input, "instruction": _instruction})
            source.append(result)
        return {"source": source}

    def add_eos_token_function(self, examples):
        return {
            "target": [
                f"{example}{self.tokenizer.eos_token}" for example in examples["output"]
            ]
        }

    def _tokenize_strings(self, strings: Sequence[str]):
        tokenized_list = [
            self.tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = [
            tokenized.input_ids.ne(self.tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=input_ids,
            input_ids_lens=input_ids_lens,
            labels_lens=input_ids_lens,
        )

    def tokenize_function(self, examples):
        sources = examples["source"]
        concatenated_examples = [
            s + t for s, t in zip(examples["source"], examples["target"])
        ]
        concatenated_examples_tokenized, sources_tokenized = [
            self._tokenize_strings(strings)
            for strings in (concatenated_examples, sources)
        ]
        input_ids = concatenated_examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = self.IGNORE_INDEX
        return {"input_ids": input_ids, "labels": labels}

    @staticmethod
    def collate_fn(
        pad_token_id: int, padding_value: int, batch: Sequence[Dict[str, Tensor]]
    ) -> Dict[str, Tensor]:
        input_ids = [example["input_ids"] for example in batch]
        labels = [example["labels"] for example in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=padding_value
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(pad_token_id),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size,
            collate_fn=partial(
                self.collate_fn, self.tokenizer.pad_token_id, self.IGNORE_INDEX
            ),
            num_workers=8,
        )
