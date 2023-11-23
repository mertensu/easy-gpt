import json
from data_module import CharTokenizer
from lightning_module import LanguageModule
from utils import load_latest_checkpoint


if __name__ == "__main__":
    keep_n_recent = 200
    max_new_tokens = 50
    n_blocks = 10

    with open("tokenizer/char_tokenizer.json", "r") as json_file:
        loaded_data = json.load(json_file)
    tkzr = CharTokenizer.from_dict(loaded_data)
    # GENERATE PATH
    ckpt_path = load_latest_checkpoint("lightning_logs")
    lang_model = LanguageModule.load_from_checkpoint(ckpt_path)
    gen_ids = lang_model.endless_generate(
        "@He was ",
        tokenizer=tkzr,
        keep_n_recent=keep_n_recent,
        max_new_tokens=max_new_tokens,
        n_blocks=n_blocks,
    )
    gen_text = tkzr.decode(gen_ids)
    print(gen_text)
