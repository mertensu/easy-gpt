import os

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from data_module import TxtDataModule, CharTokenizer
from lightning_module import LanguageModule
import json

if __name__ == "__main__":
    # LOGGING
    wanna_use_wandb = False  # future feature

    # MODEL
    context_len = 256
    batch_size = 32
    attn_dropout = 0.1
    mlp_dropout = 0.1
    d_model = 128
    n_heads = 2
    n_layers = 4
    mlp_mult_factor = 4
    max_lr = 5e-4

    # INFERENCE
    # those tokens will be added to the beginning of every sequence in this order
    special_tokens = ["@"]
    train_percentage = 0.995
    limit_val_batches = 100

    # TRAINER
    max_steps = 2_000
    check_val_every_k_steps = 500

    dm = TxtDataModule(
        txt_file_name="harry2.txt",
        data_dir="data",
        batch_size=batch_size,
        context_len=context_len,
        train_percentage=train_percentage,
        special_start_tokens=special_tokens,
    )

    # if tokenizer not available, train it
    if not os.path.exists("tokenizer/char_tokenizer.json"):
        dm.prepare_data()

    # Load pretrained tokenizer
    # to access vocab size
    with open("tokenizer/char_tokenizer.json", "r") as json_file:
        loaded_data = json.load(json_file)
    tkzr = CharTokenizer.from_dict(loaded_data)

    lang_model = LanguageModule(
        d_model=d_model,
        vocab_size=tkzr.vocab_size,
        context_len=dm.context_len,
        n_layers=n_layers,
        n_heads=n_heads,
        mlp_mult_factor=mlp_mult_factor,
        attn_dropout=attn_dropout,
        mlp_dropout=mlp_dropout,
        max_lr=max_lr,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    if wanna_use_wandb:
        wandb_logger = WandbLogger(
            name="medium_256_d_model", project="easy-gpt", log_model="all"
        )
    trainer = Trainer(
        max_steps=max_steps,
        val_check_interval=check_val_every_k_steps,
        callbacks=[lr_monitor],
        limit_val_batches=limit_val_batches,
        logger=wandb_logger if wanna_use_wandb else None,
    )

    trainer.fit(lang_model, dm)
