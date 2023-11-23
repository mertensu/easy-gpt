import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from data_module import CharTokenizer
from einops import rearrange
from typing import Tuple, List
from kv_cache import KVUpdater
from model import Transformer


def generate_causal_float_mask(seq_len):
    return torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)


def generate_special_token_mask(seq_len, n_special_tokens=None):
    mask = torch.zeros((1, seq_len))
    if n_special_tokens is not None:
        mask[:, :n_special_tokens] = float("-inf")
    return mask


class LanguageModule(pl.LightningModule):
    """
    Pytorch Lightning language module

    Args:
        d_model (int): the input/embedding dimensionality
        vocab_size (int): the size of the vocabulary, i.e. how many unique tokens
        context_len (int): the sequence length to train on, determines the max. number of tokens processed
        n_layers (int): the number of mha decoder layers
        n_heads (int): the number of heads used
        mlp_mult_factor (int): the factor by which to expand the input in the FeedForward net (default: 4)
        attn_dropout (float): dropout used in multi-head attention (default: 0.1)
        mlp_dropout (float): dropout used in Feedforward net
        max_lr (float): the maximum learning rate to reach
        n_special_tokens (int): how many special tokens should be masked (optional) default: no (None)
        lora_config (Dict): a config dict with lora rank and alpha value in case of finetuning, else None

    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        context_len: int,
        n_layers: int,
        n_heads: int,
        mlp_mult_factor: int,
        attn_dropout: float,
        mlp_dropout: float,
        max_lr: float,
        n_special_tokens=None,
        lora_config=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.max_lr = max_lr
        self.transformer = Transformer(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
            n_heads=n_heads,
            mlp_mult_factor=mlp_mult_factor,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
            lora_config=lora_config,
        )
        # construct causal attention mask
        self.register_buffer(
            "causal_attn_mask", generate_causal_float_mask(context_len)
        )
        # construct padding mask (note: default is no masking at all, only for potential future use)
        self.register_buffer(
            "special_token_mask",
            generate_special_token_mask(context_len, n_special_tokens),
        )

    def forward(
        self,
        inp_seq: torch.Tensor,
        past_key_values: Tuple[Tuple[torch.Tensor, ...], ...] = None,
    ) -> Tuple[torch.Tensor, ..., Tuple[Tuple[torch.Tensor, ...], ...]]:
        """
        Forward pass through Transformer model

        Args:
            inp_seq: tensor of shape (B, S) holding the raw tokens
            past_key_values: Tuple holding the k,v caches of each decoder layer

        Returns:
            logits, attention weights, past_key_values
        """
        return self.transformer(
            inp_seq=inp_seq,
            attn_mask=self.causal_attn_mask,
            pad_mask=self.special_token_mask,
            past_key_values=past_key_values,
        )

    def training_step(
        self,
        batch: Tuple[torch.tensor],
        batch_idx: int,
        past_key_values: Tuple[Tuple[torch.Tensor, ...], ...] = None,
    ) -> torch.Tensor:
        """
        Run forward training pass and return loss

        Args:
            batch (Tuple[torch.tensor]): one batch of data containing x = sequence and y = sequence shifted by 1
            batch_idx (int): the index of the respective batch
            past_key_values: Tuple holding the k,v caches of each decoder layer

        Returns:
            torch.Tensor: the loss for that batch
        """
        inp_seq, target_seq = batch
        # we are only interested in the logits
        logits, _, _ = self.transformer(
            inp_seq=inp_seq,
            attn_mask=self.causal_attn_mask,
            pad_mask=self.special_token_mask,
            past_key_values=past_key_values,
        )
        # we have batch_size * seq_len predictions, hence merge into one dim
        logits = rearrange(logits, "b s d -> (b s) d")
        # make a target of the same shape but implicit last dim of 1 (since only a single token is correct)
        target_seq = rearrange(target_seq, "b s -> (b s)")
        # compute loss
        loss = F.cross_entropy(logits, target_seq)
        # log the loss
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Run forward validation pass and return loss

        Args:
            batch (Tuple[torch.tensor]): one batch of data containing x = sequence and y = sequence shifted by 1
            batch_idx (int): the index of the respective batch

        Returns:
            torch.Tensor: the val loss for that batch
        """
        inp_seq, target_seq = batch
        logits, _, _ = self.transformer(
            inp_seq=inp_seq,
            attn_mask=self.causal_attn_mask,
            pad_mask=self.special_token_mask,
        )
        logits = rearrange(logits, "b s d -> (b s) d")
        target_seq = rearrange(target_seq, "b s -> (b s)")
        loss = F.cross_entropy(logits, target_seq)
        self.log("val_loss", loss, prog_bar=True, on_step=True)
        return loss

    def configure_optimizers(self) -> Tuple:
        """
        Configure optimizer and lr scheduler

        Returns:
            tuple of lists holding optimizer and scheduler
        """
        optim = torch.optim.AdamW(self.parameters(), lr=self.max_lr)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optim, max_lr=self.max_lr, total_steps=self.trainer.max_steps
        )
        return [optim], [{"scheduler": lr_scheduler, "interval": "step"}]

    @torch.no_grad()
    def endless_generate(
        self,
        text: str,
        tokenizer: CharTokenizer,
        keep_n_recent: int,
        max_new_tokens: int,
        n_blocks: int,
    ) -> List[int]:
        """
        Runs inference to predict max_new_tokens * n_blocks tokens using kv-caching and attention sinks.
        The actual idea is to enable long conversations beyond the context length trained so you can imagine
        each block in n_blocks being a prompt (question). Here it is kind of useless but I want to keep the
        structure.

        Args:
            text (str): a string as a starting point for generation
            tokenizer (CharTokenizer): a tokenizer (currently only character tokenization)
            keep_n_recent (int): how many tokens to keep in cache for predicting the next
            max_new_tokens (int): the number of tokens to generate in each block
            n_blocks (int): the number of blocks

        Returns:
            List[str]: a list containing the predicted tokens
        """
        # init the attention-sink kv-caching
        kv_updater = KVUpdater(keep_n_recent=keep_n_recent)
        self.transformer.eval()

        # map each character to a token, result list of tokens
        inp_seq = tokenizer.encode(text)
        # we get a list of tokens and convert to tensor with batch dim, result (1, S)
        # note that we need to put on device in this (only) case since text is coming
        # from outside the model and lies on cpu.
        token_idx = torch.tensor(inp_seq, dtype=torch.long)[None, ...].to(self.device)
        # pass through model and get next token prediction plus past_key_values
        logits, _, past_key_values = self.transformer(
            inp_seq=token_idx,
            attn_mask=self.causal_attn_mask,
            pad_mask=self.special_token_mask,
        )
        # get logits for last token prediction (next token)
        logits = logits[:, -1, :]
        # convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample a single token_idx from this distribution
        token_idx_next = torch.multinomial(probs, num_samples=1)
        # store in list
        generated_ids = [token_idx_next.item()]

        # Now we can start with the actual updating of kv-cache according to
        # the attention sink idea. We have some buffer (keep_n_recent) we want to
        # always use for making a prediction of the next token.
        # 1. Check how much we already store
        # 2. Determine the space we need for the next block
        # 3. If it fits into the buffer, all fine, if not, cut away some part from the beginning of the kv-cache
        # but always keep first token (attention sink). See kv_updater for further info.
        # 4. Run inference to get max_new_tokens.

        for _ in range(n_blocks):
            # how long is the sequence we already store
            already_stored = past_key_values[0][0].size(1)
            # how long will the sequence get after the next block of generating max_new_tokens
            space_needed = already_stored + max_new_tokens
            # maybe update past_key_values such that the space_needed fits into buffer
            past_key_values = kv_updater.throw_away(past_key_values, space_needed)

            # inner loop, generate k new tokens and overwrite past_key_values
            for _ in range(max_new_tokens):
                (
                    last_token_logits,
                    past_key_values,
                ) = self.get_last_token_logits_using_cache(
                    token_idx_next, past_key_values
                )
                probs = F.softmax(last_token_logits, dim=-1)
                token_idx_next = torch.multinomial(probs, num_samples=1)
                generated_ids.append(token_idx_next.item())

        return generated_ids

    def get_last_token_logits_using_cache(
        self,
        token_idx_next: torch.Tensor,
        past_key_values: Tuple[Tuple[torch.Tensor, ...], ...],
    ) -> Tuple:
        """
        Helper function, simply return the logits of the last token based on kv-cache and
        the updated kv-cache. token_idx_next should therefore be a single token only.

        Args:
            token_idx_next (torch.Tensor): the last predicted token
            past_key_values: tuple of kv-caches for each decoder layer

        Returns:
            the logits of each token being the next one, update kv-cache
        """
        logits, _, pkv = self.transformer(
            inp_seq=token_idx_next,
            attn_mask=self.causal_attn_mask,
            pad_mask=self.special_token_mask,
            past_key_values=past_key_values,
        )
        return logits[:, -1, :], pkv

    @torch.no_grad()
    def generate(self, text, tokenizer, max_new_tokens=30, use_past=False):
        self.transformer.eval()

        inp_seq = tokenizer.encode(text)
        # we get a list of tokens and convert to tensor with batch dim
        token_idx = torch.tensor(inp_seq, dtype=torch.long)[None, ...]
        logits, _, past_key_values = self.transformer(
            inp_seq=token_idx,
            attn_mask=self.causal_attn_mask,
            pad_mask=self.special_token_mask,
        )
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        token_idx_next = torch.multinomial(probs, num_samples=1)
        token_idx = torch.cat((token_idx, token_idx_next), dim=1)
        generated_ids = [token_idx_next.item()]
        for i in range(max_new_tokens):
            if use_past:
                token_idx = token_idx_next
            out = self.transformer(
                inp_seq=token_idx,
                attn_mask=self.causal_attn_mask,
                pad_mask=self.special_token_mask,
                past_key_values=past_key_values if use_past else None,
            )
            logits = out[0]
            past_key_values = out[2]
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            token_idx_next = torch.multinomial(probs, num_samples=1)
            generated_ids.append(token_idx_next.item())
            if not use_past:
                token_idx = torch.cat((token_idx, token_idx_next), dim=1)

        return generated_ids
