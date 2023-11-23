# easy-gpt

This is a repo containing well-documented pytorch code for training a 
simple GPT-like (transformer decoder) model on some txt-file of your choice using character tokenization.

It also contains simplified implementations of some recent advances

- Rotary Position Embedding (https://arxiv.org/abs/2104.09864)
- Attention Sinks (https://arxiv.org/abs/2309.17453)
- 
The code is mainly for teaching-purposes. It is not written to be fast and performant but to be as
readable as possible. 

All model-related details are plain Python/Pytorch; the training is simplified
using Pytorch-Lightning (https://lightning.ai/).

## How to use

### TRAINING

Please use Python **3.11** and install the requirements first:
```
pip install -r requirements.txt
```

Next, you can either just train a model on the data (Harry Potter) that have already been
preprocessed (see `data/preprocessed`) by running

```
python run.py
```

Alternatively, you can train on your own txt-file. Please put the file into the data directory and
adjust the name in `run.py`

```
dm = TxtDataModule(
        txt_file_name="your_file_name.txt",
        ...
```

It will write back `train.bin` and `val.bin` and hence overwrite the existing files if you do not
change their names in `data_module.py`

```
data_bin_fp = f"{self.data_dir}/preprocessed/{stage}.bin"
```

### INFERENCE

Once you trained a model, the checkpoint will be automatically stored in `lightning_logs` under
a specific version directory. You can then call 

```
python generate.py
```

to generate new tokens based on the most recent run and some starting prompt. Feel free to adjust `generate.py` to
your needs. 

## Future ideas:

- finetuning (there is already a `CustomLinearLayer`) 
- other tokenizers (such as BPE)
- more advanced sampling techniques
- ...