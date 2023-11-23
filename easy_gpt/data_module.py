from __future__ import annotations


import logging

import lightning.pytorch as pl
import torch
from utils import remove_unicode_chars
import numpy as np
import os
import json
from typing import List, Tuple, Dict
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger(__name__)


class CharTokenizer:
    """
    Character tokenizer

    Args:
        name (str): name of the tokenizer
        special_tokens (List): the special tokens used
        vocab_size (int): number of unique tokens
        stoi (Dict): dict mapping from string to index (encoding)
        itos (Dict): dict mapping from index to string (decoding)
        special_token_idx (List): list of indices of special tokens
        trained (bool): whether the tokenizer has already been trained
    """

    def __init__(
        self,
        name: str,
        special_tokens: List[str],
        vocab_size: int = None,
        stoi: Dict[str, int] = None,
        itos: Dict[int, str] = None,
        special_token_idx: List[int] = None,
        trained: bool = False,
    ):
        super().__init__()
        self.name = name
        self.special_tokens = special_tokens
        self.vocab_size = vocab_size
        self.stoi = stoi
        self.itos = itos
        if special_token_idx is None:
            self.special_token_idx = []
        else:
            self.special_token_idx = special_token_idx
        self.trained = trained

    def train(self, data: str) -> None:
        """
        Compute vocabulary size, mapping from string to index and vice versa,
        and handle special tokens.

        Args:
            data (str): string holding some text

        Returns:
            None
        """
        # get a sorted list of all characters present in text
        chars = sorted(list(set(data)))
        # get vocabulary size as number of unique characters
        self.vocab_size = len(chars)
        # dict from character to index for encoding
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        # dict from index to character for decoding
        self.itos = {i: ch for i, ch in enumerate(chars)}
        # prepend special tokens
        self.add_special_tokens()
        # set trained to True
        self.trained = True

    def add_special_tokens(self) -> None:
        """
        Append special tokens to vocabulary and returns indices of special tokens

        Returns:
            None
        """
        # increase the vocab size by the number of special tokens
        self.vocab_size += len(self.special_tokens)
        # get the index to store the first special token
        start_idx = len(self.stoi)
        # loop over special tokens and append to encoding/decoding dicts
        for i, special_token in enumerate(self.special_tokens):
            self.stoi[special_token] = start_idx + i
            self.itos[start_idx + i] = special_token
            self.special_token_idx.append(start_idx + i)

    def encode(self, text: str) -> List[int]:
        """
        Convert each token in a string to its index in the vocabulary

        Args:
            text (str): string holding some text

        Returns:
            List[int]: list of indices
        """
        return [self.stoi[c] for c in text]

    def decode(self, indices: List[int]) -> str:
        """
        Map each index in a list to its corresponding character in the vocabulary and convert to string

        Args:
            indices (List[int]): list of indices

        Returns:
            str: string of characters corresponding to passed indices
        """
        return "".join([self.itos[i] for i in indices])

    def to_dict(self) -> Dict:
        """
        Fill all data relevant for later use in dictionary

        Returns:
            Dict: all info regarding the trained tokenizer
        """
        return {
            "name": self.name,
            "vocab_size": self.vocab_size,
            "stoi": self.stoi,
            "itos": self.itos,
            "special_token_idx": self.special_token_idx,
            "special_tokens": self.special_tokens,
            "trained": self.trained,
        }

    @classmethod
    def from_dict(cls, data_dict: Dict) -> CharTokenizer:
        """
        Initialize class from dictionary (used when loading the pretrained tokenizer)

        Args:
            data_dict: dict containing all necessary information for initialization

        Returns:
            CharTokenizer: the initialized class instance
        """
        assert data_dict["trained"], "Tokenizer has not been trained yet"
        data_dict["itos"] = {int(k): v for k, v in data_dict["itos"].items()}
        return cls(**data_dict)

    def to_json_string(self) -> str:
        """
        Convert dict to json string.

        Returns:
            str
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: str) -> None:
        """
        Store json string to file

        Args:
            json_file_path: the complete path where the file should be stored

        Returns:
            None
        """
        """Save this instance to a json file."""
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())


class TxtDataset(IterableDataset):
    """
    Pytorch Dataset for text data. Code from lit-gpt (with slight adjustments)
    see: https://github.com/Lightning-AI/lit-gpt.

    NOTE: It is only able to handle special tokens that are added to the beginning of every training sample
    in order to apply attention sinks (https://github.com/mit-han-lab/streaming-llm).

    Args:
        data_file (str): path to data-file in binary format
        context_len (int): the amount of tokens to train on
        special_token_idx (List): the list of indices of the special tokens
    """

    def __init__(
        self, data_file: str, context_len: int, special_token_idx: List[int] = None
    ):
        super().__init__()
        self.data_file = data_file
        self.context_len = context_len
        self.special_token_idx = special_token_idx
        self.n_special_tokens = len(self.special_token_idx)
        self.special_token_tensor = torch.tensor(self.special_token_idx)

    def __iter__(self):
        """
        Generate a training sample and the corresponding target

        Returns:
            Tuple[torch.Tensor,torch.Tensor]: tensor of indices, tensor of indices shifted by 1
        """
        # load data in memory map
        data = np.memmap(self.data_file, dtype=np.uint16, mode="r")
        while True:
            # pick a random index, last possible one should still leave enough space to fit context_len indices
            idx = torch.randint(len(data) - self.context_len, (1,)).item()
            # we want to have a sequence length of context_len to train on.
            # However, we need to reserve some space for the special tokens.
            x = torch.from_numpy(
                (data[idx : (idx + (self.context_len - self.n_special_tokens))]).astype(
                    np.int64
                )
            )
            # special tokens should not be involved in shifting, i.e. a special token
            # as input should predict the same special token

            # y = ['<special>', 'e', 'l', 'l', 'o',]
            # x = ['<special>', 'H', 'e', 'l', 'l',]

            y = torch.from_numpy(
                (
                    data[
                        idx + 1 : (idx + 1 + (self.context_len - self.n_special_tokens))
                    ]
                ).astype(np.int64)
            )
            if self.n_special_tokens > 0:
                # iterate over list of special tokens in reverse order
                # TODO: Is there a better way? Having to adjust every sample is not nice.
                for i in reversed(range(self.n_special_tokens)):
                    x = torch.cat((self.special_token_tensor[[i]], x))
                    y = torch.cat((self.special_token_tensor[[i]], y))
            yield x, y


class TxtDataModule(pl.LightningDataModule):
    """
    A lightning data module for preparing text data for autoregressive language modeling

    Args:
        txt_file_name (str): name of txt-file containing the data to train the model on
        self.data_dir (str): directory where the preprocessed/tokenized data should be stored
        batch_size (int): the batch-size used
        context_len (int): the sequence length to train on, determines the max. number of tokens processed
        train_percentage (float): how much percent of the dataset should be used for training
        special_start_tokens (List[str]): a list of special start tokens ordered as they should be prepended

    """

    def __init__(
        self,
        txt_file_name: str,
        data_dir: str,
        batch_size: int,
        context_len: int,
        train_percentage: float,
        special_start_tokens: List[str] = None,
    ):
        super().__init__()
        self.txt_file_name = txt_file_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.context_len = context_len
        self.train_percentage = train_percentage
        self.special_tokens = special_start_tokens
        self.charTokenizer = CharTokenizer(
            name="charTokenizer", special_tokens=self.special_tokens
        )

    def prepare_data(self) -> None:
        """
        Read text file, tokenize and store as pkl for later use.

        Returns:
            None
        """
        if not os.path.exists("tokenizer/char_tokenizer.json"):
            # open txt file on which you would like to train the model
            with open(
                f"{self.data_dir}/{self.txt_file_name}", "r", encoding="ISO-8859-1"
            ) as f:
                text = f.read()
                # replace newline character with white space
                text = text.replace("\n", " ")
                # only needed to remove some unicode chars such that the vocab looks nicer
                # ,i.e. contains only regular characters
                text = remove_unicode_chars(text)
            # construct vocabulary and dict mappings
            self.charTokenizer.train(text)
            logger.info("CharTokenizer trained")
            # store trained tokenizer
            self.charTokenizer.to_json_file("tokenizer/char_tokenizer.json")
            # tokenize input data
            tokenized_text = np.array(self.charTokenizer.encode(text), dtype=np.uint16)

            # compute first token used in validation sete
            n = int(self.train_percentage * len(tokenized_text))
            # store training set in memory map
            self._store_in_memmap(tokenized_text[:n], stage="train")
            # store validation set in memory map
            self._store_in_memmap(tokenized_text[n:], stage="val")
        else:
            logger.info("Data preparation already done, skipping...")
            pass

    def _store_in_memmap(self, tokenized_text: np.ndarray, stage: str) -> None:
        """
        Stores tokenized text in binary format for later use

        Args:
            tokenized_text (np.ndarray): array holding the indices of the tokens
            stage: either 'train' or 'val'

        Returns:
            None
        """
        # Create a memory-mapped array
        data_bin_fp = f"{self.data_dir}/preprocessed/{stage}.bin"
        if os.path.exists(data_bin_fp):
            # If the file exists, remove it
            os.remove(data_bin_fp)
        arr = np.memmap(
            data_bin_fp, dtype="uint16", mode="w+", shape=len(tokenized_text)
        )
        # write
        arr[:] = tokenized_text

    def setup(self, stage) -> None:
        """
        Load the encoded pickle file and construct Pytorch dataset for training and validation

        Args:
            stage (str): either train or val (currently only used once)

        Returns:
            None
        """
        logger.info("Loading pretrained tokenizer and initializing torch datasets")
        # Load pretrained tokenizer to get access to special_token_idx
        with open("tokenizer/char_tokenizer.json", "r") as json_file:
            loaded_data = json.load(json_file)
        tkzr = CharTokenizer.from_dict(loaded_data)

        self.ds_train = TxtDataset(
            f"{self.data_dir}/preprocessed/train.bin",
            context_len=self.context_len,
            special_token_idx=tkzr.special_token_idx,
        )
        self.ds_val = TxtDataset(
            f"{self.data_dir}/preprocessed/val.bin",
            context_len=self.context_len,
            special_token_idx=tkzr.special_token_idx,
        )

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size)
