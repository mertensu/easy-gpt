from utils import load_latest_checkpoint
from data_module import FinetuningDataset, BytePairTokenizer, CharTokenizer
from lightning_module import LanguageModule
from lightning.pytorch import Trainer
from torchinfo import summary


import json
from torch.utils.data import random_split, DataLoader


if __name__ == "__main__":
    # TRAINING
    batch_size = 32
    train_percentage = 0.90

    # FINETUNE
    # just random values
    lora_config = {"rank": 3, "alpha": 1}
    max_prompt_len_prop = 0.7

    # trainer args
    max_steps = 100

    ckpt_path = load_latest_checkpoint("lightning_logs")

    # load pretrained model and set lora_config
    lang_model = LanguageModule.load_from_checkpoint(
        ckpt_path, lora_config=lora_config, strict=False
    )

    use_bpe = True

    if use_bpe:
        # Load pretrained tokenizer to access vocab size
        with open("tokenizer/bpeTokenizer.json", "r") as json_file:
            loaded_data = json.load(json_file)
            tokenizer = BytePairTokenizer.from_dict(loaded_data)
    else:
        with open("tokenizer/charTokenizer.json", "r") as json_file:
            loaded_data = json.load(json_file)
            tokenizer = CharTokenizer.from_dict(loaded_data)

    # alternatively, import your own data here
    with open("data/harry_finetuning.json", "r") as json_file:
        data = json.load(json_file)

    # has to stay like this for now, do not change context_len
    context_len = lang_model.hparams.get("context_len")

    finetuning_dataset = FinetuningDataset(
        data=data,
        prompt_col_name="question",
        response_col_name="answer",
        tokenizer=tokenizer,
        context_len=context_len,
        max_prompt_len_prop=max_prompt_len_prop,
    )

    ds_train, ds_val = random_split(
        finetuning_dataset, [train_percentage, 1 - train_percentage]
    )
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    test_input_tensor, _ = next(iter(train_loader))
    summary(
        lang_model,
        input_data=test_input_tensor,
        depth=4,
        col_names=["output_size", "num_params", "trainable"],
    )

    trainer = Trainer(max_steps=max_steps)
    trainer.fit(
        lang_model, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )
