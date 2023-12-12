
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import glob
from data_module import TxtDataModule, CharTokenizer, BytePairTokenizer
from lightning_module import LanguageModule
import json

if __name__ == "__main__":
    # LOGGING
    wanna_use_wandb = False  # future feature

    txt_file_name = "harry2.txt"

    # TOKENIZER
    # use byte-pair encoding, else character tokenization
    use_bpe = True

    # MODEL
    context_len = 64
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
    max_steps = 500
    check_val_every_k_steps = 200

    dm = TxtDataModule(
        data_dir="data",
        batch_size=batch_size,
        context_len=context_len,
        train_percentage=train_percentage,
        special_start_tokens=special_tokens,
    )

    if use_bpe:
        # if tokenizer not available, train it on your own data
        tokenizer = BytePairTokenizer(
            name="bpeTokenizer",
            file_path=f"data/{txt_file_name}",
            num_merges=500,
            special_tokens=special_tokens,
        )
    else:
        tokenizer = CharTokenizer(name="charTokenizer", special_tokens=special_tokens)

    if not glob.glob(f"tokenizer/{tokenizer.name}.json"):
        if txt_file_name is None:
            raise ValueError(
                "You want to train on your own txt file but no name is provided"
            )
        dm.tokenize_and_store(tokenizer=tokenizer, txt_file_name=txt_file_name)

    # Load pretrained tokenizer to access vocab size
    with open(f"tokenizer/{tokenizer.name}.json", "r") as json_file:
        loaded_data = json.load(json_file)
    if use_bpe:
        tokenizer = BytePairTokenizer.from_dict(loaded_data)
    else:
        tokenizer = CharTokenizer.from_dict(loaded_data)

    dm.build_datasets(tokenizer)

    lang_model = LanguageModule(
        d_model=d_model,
        vocab_size=tokenizer.vocab_size,
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
        wandb_logger = WandbLogger(name="small", project="easy-gpt", log_model="all")
    trainer = Trainer(
        max_steps=max_steps,
        val_check_interval=check_val_every_k_steps,
        callbacks=[lr_monitor],
        limit_val_batches=limit_val_batches,
        logger=wandb_logger if wanna_use_wandb else None,
    )

    trainer.fit(lang_model, dm)
