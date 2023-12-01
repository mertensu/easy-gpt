from __future__ import annotations


import logging

import lightning.pytorch as pl
import torch
from utils import remove_unicode_chars
import numpy as np
import os
import json
import re
import collections
from typing import List, Tuple, Dict
from torch.utils.data import DataLoader, IterableDataset
import glob

logger = logging.getLogger(__name__)


class BytePairTokenizer:
    """
    Byte Pair Encoding (adjusted from https://leimao.github.io/blog/Byte-Pair-Encoding/)

    Args:
        name (str): name of the tokenizer
        special_tokens (List): the special tokens used
        vocab_size (int): number of unique tokens
        token2index (Dict): dict mapping from toen to index (encoding)
        index2token (Dict): dict mapping from index to token (decoding)
        word2subwords (Dict): dict mapping from word to tokens it consists of
        special_token_idx (List): list of indices of special tokens
        trained (bool): whether the tokenizer has already been trained
        file_path (str): the file it is trained on
        num_merges (int): the number of merges (default 1000)

    """

    def __init__(
        self,
        name: str,
        special_tokens: List[str],
        vocab_size: int = None,
        token2index: Dict[str, int] = None,
        index2token: Dict[int, str] = None,
        word2subwords: Dict[str, List[str]] = None,
        file_path: str = None,
        num_merges: int = 1000,
        special_token_idx: List[int] = None,
        trained: bool = False,
    ):
        self.name = name
        self.file_path = file_path
        self.num_merges = num_merges
        self.special_tokens = special_tokens
        self.token2index = token2index
        self.index2token = index2token
        self.word2subwords = word2subwords
        self.vocab_size = vocab_size
        self.trained = trained
        if special_token_idx is None:
            self.special_token_idx = []
        else:
            self.special_token_idx = special_token_idx

    def train(self, data) -> None:
        """
        Train BPE tokenizer

        Args:
            data: just placeholder to have same args as CharTokenizer.train

        Returns:
            None
        """
        # consturct dictionary holding words split by character/token
        word_dict = self.word2freq()
        for i in range(self.num_merges):
            # compute dictionary of pairs
            pairs = self.pair2freq(word_dict=word_dict)
            if not pairs:
                break
            # get most occuring pair
            best_pair = max(pairs, key=pairs.get)
            # merge/melt all tokens such that only this pair exists
            word_dict = self.merge(best_pair, word_dict_old=word_dict)
            # iterate over words and store indivual tokens in dict
            # Example 'H e l lo' -> [H], [e], [l], [lo]
            token_dict = self.token2freq(word_dict)

        # construct mapping from token to index
        self.token2index = {token: i for i, token in enumerate(token_dict.keys())}
        # reverse
        self.index2token = {i: token for i, token in enumerate(token_dict.keys())}
        # number of tokens is vocab size
        self.vocab_size = len(self.token2index)
        # set to trained
        self.trained = True
        # construct dict mapping from original word to the tokens it consists of
        self.word2subwords = self.precompute_subwords(word_dict)
        # add special tokens
        self.add_special_tokens()

    def add_special_tokens(self) -> None:
        """
        Append special tokens to vocabulary and returns indices of special tokens

        Returns:
            None
        """
        # increase the vocab size by the number of special tokens
        self.vocab_size += len(self.special_tokens)
        # get the index to store the first special token
        start_idx = len(self.token2index)
        # loop over special tokens and append to encoding/decoding dicts
        for i, special_token in enumerate(self.special_tokens):
            self.token2index[special_token + '</w>'] = start_idx + i
            self.index2token[start_idx + i] = special_token + '</w>'
            self.special_token_idx.append(start_idx + i)
            self.word2subwords[special_token] = [special_token + '</w>']

    def prepare_for_encoding(self, word: str) -> str:
        """
        Prepare a word for encoding

        Args:
            word: a word

        Returns:
            str: the preprocessed word
        """
        # Insert </w> at every whitespace
        modified_word = word.replace(" ", "</w>")
        # replace newline with </w>
        modified_word = modified_word.replace("\n", "</w>")
        # add </w> at end of string
        modified_word += "</w>"
        return modified_word

    def prepare_for_presenting(self, text: str) -> str:
        """
        Prepare some text for presenting.

        Args:
            text (str): a piece of text

        Returns:
            str: the text as shown to the user
        """
        modified_text = text.replace("</w>", " ")
        return modified_text.rstrip()

    def encode(self, text: str) -> List[int]:
        """
        Encode some text, i.e. map tokens to indices.

        Args:
            text (str): a piece of text

        Returns:
            List[int]: the indices corresponding to the tokens
        """
        indices = []
        # iterate over every word in the text
        for word in text.strip().split():
            # if word is a key in the dict
            if word in self.word2subwords:
                # get the tokens the word consists of
                tokens = self.word2subwords[word]
                # extend the list
                indices.extend([self.token2index[token] for token in tokens])
            # if word is not in the dict
            else:
                # preprocess
                preprocessed_word = self.prepare_for_encoding(word)
                # tokenize
                tokens = self.tokenize(preprocessed_word)
                # extend the list
                indices.extend([self.token2index[token] for token in tokens])
        return indices

    def decode(self, indices: List[int]) -> str:
        """
        Decode some text, i.e., map from indices to tokens and reconstruct
        the original words for presenting.

        Args:
            indices (List[int]): the list of indices corresponding to tokens

        Returns:
            str: The decoded sentence
        """
        decoded_tokens = [self.index2token[index] for index in indices]
        text = "".join(decoded_tokens)
        return self.prepare_for_presenting(text)

    def tokenize(self, word: str) -> List[str]:
        """
        Split a word into tokens.

        Args:
            word: a word

        Returns:
            List[str]: the list of tokens the word consists of
        """

        if word in self.token2index:
            # If the token is in the dict of tokens, return it as a single subword unit
            return [word]
        else:
            # If the token is not, break it into subword units
            tokens = []
            start = 0
            while start < len(word):
                # Find the longest subword unit in the vocabulary
                end = len(word)
                while end > start:
                    token = word[start:end]
                    if token in self.token2index:
                        tokens.append(token)
                        start = end
                        break
                    end -= 1
            return tokens

    def word2freq(self) -> Dict[str, int]:
        """
        Compute for each word in a file the frequency it occurs

        Returns:
            Dict[str, int]: dict mapping from word to frequency
        """
        word_dict = collections.defaultdict(int)

        with open(self.file_path, "r", encoding="ISO-8859-1") as file:
            for line in file:
                words = line.strip().split()
                for word in words:
                    word_dict[" ".join(list(word)) + " </w>"] += 1

        return word_dict

    def pair2freq(self, word_dict: Dict[str, int]) -> Dict[str, int]:
        """
        Compute for each pair of tokens its frequency
        Example:
            [H e l lo] -> [H, e]: 5; [e, l]: 3; [l, lo]: 7

        Args:
            word_dict: dict mapping from word to frequency

        Returns:
            Dict[str, int]: dict mapping from tokenpairs to frequency
        """
        pairs = collections.defaultdict(int)
        for word, freq in word_dict.items():
            chars = word.split()
            for i in range(len(chars) - 1):
                pairs[chars[i], chars[i + 1]] += freq
        return pairs

    def merge(self, pair: str, word_dict_old: Dict[str, int]) -> Dict[str, int]:
        """
        Construct new token from pair and update word dict.

        Example:
        'H e l l o' best_pair: 'lo' -> 'H e l lo'

        Args:
            pair (str): pair of chars/tokens
            word_dict_old: dict mapping from word to frequency

        Returns:
            Dict[str, int]: dict mapping from word to frequency
        """
        word_dict_new = {}
        bigram = re.escape(" ".join(pair))
        # (?<!\S) asserts that there is no non-whitespace character
        # (\S) immediately before the current position in the string.
        # In other words, it checks for a position in the string where
        # the preceding character is either a whitespace character
        # or the beginning of the string.
        p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        for word in word_dict_old:
            # replace pattern matching p with the joined pair in the word
            w_out = p.sub("".join(pair), word)
            # give the newly created string the same count as the original one
            word_dict_new[w_out] = word_dict_old[word]
        return word_dict_new

    def token2freq(self, word_dict: Dict[str, int]) -> Dict[str, int]:
        """
        Build mapping from token to its frequency

        Args:
            word_dict: dict mapping from word to frequency

        Returns:
            Dict[str, int]: mapping from token to frequency
        """
        token_dict = collections.defaultdict(int)
        for word, freq in word_dict.items():
            word_tokens = word.split()
            for token in word_tokens:
                token_dict[token] += freq
        return token_dict

    def precompute_subwords(self, word_dict: Dict[str, int]) -> Dict[str, List[str]]:
        """
        Precompute for each word the tokens it consists of for faster encoding
        Example:
            'H e l lo' -> ['H', 'e', 'l', 'lo']

        Args:
            word_dict: mapping from word to frequency

        Returns:
            Dict[str, List[str]]: mapping from word to tokens
        """
        subword_dict = collections.defaultdict(list)
        for word, freq in word_dict.items():
            word_tokens = word.split()
            # reconstruct whole word
            original_word = word.replace(" ", "")
            original_word = original_word.replace("</w>", "")
            # if word not already exists
            if not original_word in subword_dict:
                for token in word_tokens:
                    subword_dict[original_word].append(token)
        return subword_dict

    def to_dict(self) -> Dict:
        """
        Fill all data relevant for later use in dictionary

        Returns:
            Dict: all info regarding the trained tokenizer
        """
        return {
            "name": self.name,
            "vocab_size": self.vocab_size,
            "token2index": self.token2index,
            "index2token": self.index2token,
            "word2subwords": self.word2subwords,
            "special_token_idx": self.special_token_idx,
            "special_tokens": self.special_tokens,
            "file_path": self.file_path,
            "trained": self.trained,
        }

    @classmethod
    def from_dict(cls, data_dict: Dict) -> BytePairTokenizer:
        """
        Initialize class from dictionary (used when loading the pretrained tokenizer)

        Args:
            data_dict: dict containing all necessary information for initialization

        Returns:
            CharTokenizer: the initialized class instance
        """
        assert data_dict["trained"], "Tokenizer has not been trained yet"
        data_dict["index2token"] = {
            int(k): v for k, v in data_dict["index2token"].items()
        }
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
        self.data_dir (str): directory where the preprocessed/tokenized data should be stored
        batch_size (int): the batch-size used
        context_len (int): the sequence length to train on, determines the max. number of tokens processed
        train_percentage (float): how much percent of the dataset should be used for training
        special_start_tokens (List[str]): a list of special start tokens ordered as they should be prepended

    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        context_len: int,
        train_percentage: float,
        special_start_tokens: List[str] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.context_len = context_len
        self.train_percentage = train_percentage
        self.special_tokens = special_start_tokens

    def tokenize_and_store(self, tokenizer, txt_file_name) -> None:
        """
        Read text file, tokenize and store as pkl for later use.

        Returns:
            None
        """
        if not glob.glob("tokenizer/*.json"):
            # open txt file on which you would like to train the model
            with open(
                f"{self.data_dir}/{txt_file_name}", "r", encoding="ISO-8859-1"
            ) as f:
                text = f.read()
                # replace newline character with white space
                text = text.replace("\n", " ")
                # only needed to remove some unicode chars such that the vocab looks nicer
                # ,i.e. contains only regular characters
                text = remove_unicode_chars(text)
            # construct vocabulary and dict mappings
            tokenizer.train(text)
            logger.info("Tokenizer trained")
            # store trained tokenizer
            tokenizer.to_json_file(f"tokenizer/{tokenizer.name}.json")
            # tokenize input data
            tokenized_text = np.array(tokenizer.encode(text), dtype=np.uint16)
            logger.info("All data encoded")

            # compute first token used in validation sete
            n = int(self.train_percentage * len(tokenized_text))
            # store training set in memory map
            self._store_in_memmap(tokenizer.name, tokenized_text[:n], stage="train")
            # store validation set in memory map
            self._store_in_memmap(tokenizer.name, tokenized_text[n:], stage="val")
        else:
            logger.info("Data preparation already done, skipping...")
            pass

    def _store_in_memmap(self, tokenizer_name, tokenized_text: np.ndarray, stage: str) -> None:
        """
        Stores tokenized text in binary format for later use

        Args:
            tokenizer_name (str): the name of the tokenizer
            tokenized_text (np.ndarray): array holding the indices of the tokens
            stage: either 'train' or 'val'

        Returns:
            None
        """
        # Create a memory-mapped array
        data_bin_fp = f"{self.data_dir}/preprocessed/{stage}_{tokenizer_name}.bin"
        if os.path.exists(data_bin_fp):
            # If the file exists, remove it
            os.remove(data_bin_fp)
        arr = np.memmap(
            data_bin_fp, dtype="uint16", mode="w+", shape=len(tokenized_text)
        )
        # write
        arr[:] = tokenized_text

    def build_datasets(self, tokenizer) -> None:
        """
        Construct Pytorch datasets for training and validation

        Args:
            stage (str): either train or val (currently only used once)

        Returns:
            None
        """

        self.ds_train = TxtDataset(
            f"{self.data_dir}/preprocessed/train_{tokenizer.name}.bin",
            context_len=self.context_len,
            special_token_idx=tokenizer.special_token_idx,
        )
        self.ds_val = TxtDataset(
            f"{self.data_dir}/preprocessed/val_{tokenizer.name}.bin",
            context_len=self.context_len,
            special_token_idx=tokenizer.special_token_idx,
        )

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size)
