import math
from pathlib import Path
from typing import Any, Union

import transformers
from deepspeed.ops.adam import FusedAdam
from lightning import LightningModule
from transformers import (
    PreTrainedTokenizer,
    LlamaForCausalLM,
    LlamaConfig,
    AutoModelForCausalLM,
)
from transformers.optimization import get_cosine_schedule_with_warmup

from alpaca.utils import enable_transformers_pretrained_deepspeed_sharding


def get_llama_tokenizer(
    pretrained_model_name: str, model_max_length: int
) -> transformers.LlamaTokenizer:
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        pretrained_model_name,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.add_special_tokens(
        {"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"}
    )
    return tokenizer


class AlpacaLightningModule(LightningModule):
    def __init__(
        self,
        pretrained_model_name: str,
        tokenizer: PreTrainedTokenizer,
        learning_rate: float,
        weight_decay: float,
        warmup_ratio: float,
        load_weights_from_hf_pretrained_model: bool = False,
    ):
        super().__init__()

        # save parameters
        self.save_hyperparameters()

        # init model
        self.model = None
        # self.initialize_model()
        self.tokenizer = self.hparams.tokenizer
        self._hf_pipeline = None

    def setup(self, stage: str) -> None:
        # enable deepspeed sharding
        enable_transformers_pretrained_deepspeed_sharding(self)
        self.initialize_model()

    def initialize_model(self):
        if self.hparams.load_weights_from_hf_pretrained_model:
            model = LlamaForCausalLM.from_pretrained(self.hparams.pretrained_model_name)
        else:
            config = LlamaConfig.from_pretrained(self.hparams.pretrained_model_name)
            model = AutoModelForCausalLM.from_config(config=config)
        self.model = model

    def on_fit_start(self) -> None:
        # resize model
        self.model.resize_token_embeddings(len(self.tokenizer))

    def _step(self, batch):
        outputs = self(batch)
        loss = outputs.loss
        return loss

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = FusedAdam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        warmup_steps = math.ceil(
            self.hparams.warmup_ratio * self.trainer.estimated_stepping_batches
        )

        lr_schedular = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_schedular,
            "interval": "step",
        }

    def generate(self, text: str, **kwargs) -> Any:
        inputs = self.tokenizer(text, return_tensors="pt")
        return self.model.generate(inputs["input_ids"], **kwargs)

    def save_hf_checkpoint(self, path: Union[str, Path]) -> None:
        self.model.save_pretrained(path)
