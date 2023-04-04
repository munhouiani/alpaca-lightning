import mlflow.pytorch
from lightning.pytorch import Trainer, seed_everything

from alpaca.dataset import AlpacaLightningDataModule
from alpaca.model import AlpacaLightningModule, get_llama_tokenizer

if __name__ == "__main__":
    dataset_path = "data/alpaca_data.json"
    pretrained_model_name = "decapoda-research/llama-7b-hf"
    model_max_length = 512
    learning_rate = 2e-5
    weight_decay = 0
    warmup_ratio = 0.03
    load_weights_from_hf_pretrained_model = True
    batch_size = 128
    max_epochs = 3

    # seed for reproducibility
    seed_everything(42)

    # mlflow logging path
    mlflow.set_tracking_uri("file:./alpaca_mlruns")

    tokenizer = get_llama_tokenizer(
        pretrained_model_name=pretrained_model_name,
        model_max_length=model_max_length,
    )

    alpaca_datamodule = AlpacaLightningDataModule(
        data_path=dataset_path, tokenizer=tokenizer, batch_size=batch_size
    )

    alpaca_model = AlpacaLightningModule(
        pretrained_model_name=pretrained_model_name,
        tokenizer=tokenizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        load_weights_from_hf_pretrained_model=load_weights_from_hf_pretrained_model,
    )

    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=max_epochs,
        deterministic=True,
        strategy="fsdp",
        precision="16-mixed",
    )

    # Auto log all Mlflow entities
    mlflow.pytorch.autolog()

    with mlflow.start_run(run_name="alpaca_fine_tuning") as run:
        trainer.fit(alpaca_model, alpaca_datamodule)

    alpaca_model.save_hf_checkpoint("alpaca_model_huggingface_checkpoint")
    trainer.save_checkpoint("alpaca_model_lightning_checkpoint")
    mlflow.log_artifact(
        "alpaca_model_huggingface_checkpoint", "alpaca_model_huggingface_checkpoint"
    )
    mlflow.log_artifact(
        "alpaca_model_lightning_checkpoint", "alpaca_model_lightning_checkpoint"
    )
