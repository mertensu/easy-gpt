import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from einops import rearrange
import logging
from rope import RoPE
from typing import Tuple, Dict
import math

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def _get_clones(module: nn.Module, n_copies: int) -> nn.ModuleList:
    """
    Creates a list of Pytorch modules of size N

    Args:
        module (nn.Module): a pytorch module
        n_copies (int): the number of copies
    Returns:
        nn.ModuleList: A list of pytorch modules to iterate over
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n_copies)])


def prepare_attention_mask(attn_mask: torch.Tensor) -> torch.Tensor:
    """
    Checks and rearranges the attention mask.

    Args:
        attn_mask (torch.Tensor): tensor of shape (S, S)
    Returns:
        torch.Tensor: updated attention mask
    """
    if not torch.is_floating_point(attn_mask):
        raise ValueError("Please provide the mask as a float tensor")
    # causal mask should be 2d tensor (S, S), note si and sj are the same here
    return rearrange(attn_mask, "si sj -> 1 si sj")


def prepare_padding_mask(pad_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Checks and rearranges the padding mask.

    Args:
        pad_mask (torch.Tensor): raw padding mask of shape (B, S)

    Returns:
        torch.Tensor: updated padding mask
    """
    # add dimension
    return rearrange(pad_mask, "b s -> b 1 s")


class CustomLinear(nn.Module):
    """
    Custom Linear layer to enable lora-finetuning on a pretrained model

    Args:
        in_features (int): the number of input features
        out_features (int): the number of output features
        lora_config (dict): during pretraining None, otherwise rank and alpha have to be set
    """

    def __init__(self, in_features: int, out_features: int, lora_config=None):
        super().__init__()
        # during pretraining, we set rank and alpha to None
        if lora_config is None:
            self.lora_config = {"rank": None, "alpha": None}
        else:
            self.lora_config = lora_config
        self.in_features = in_features
        self.out_features = out_features
        # check whether lora is enabled
        self.lora_enabled = False if self.lora_config["rank"] is None else True

        # set A and B only when finetuning and make trainable
        if self.lora_enabled:
            self.lora_A = nn.Parameter(
                torch.zeros((self.lora_config["rank"], in_features)),
                requires_grad=self.lora_enabled,
            )
            self.lora_B = nn.Parameter(
                torch.zeros((out_features, self.lora_config["rank"])),
                requires_grad=self.lora_enabled,
            )
        # usual linear layer weight matrix, frozen during finetuning
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features)),
            requires_grad=not self.lora_enabled,
        )
        # init matrices accordingly
        self.reset_parameters(lora_enabled=self.lora_enabled)

    def reset_parameters(self, lora_enabled: bool) -> None:
        """
        Parameter initialization

        Args:
            lora_enabled (bool): whether loRA finetuning is enabled

        Returns:
            None
        """
        if not lora_enabled:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through custom linear layer

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        # call linear layer using weight
        main_out = F.linear(x, self.weight)
        # in case of pretraining, just return result
        if not self.lora_enabled:
            return main_out
        # in case of finetuning, add lora linear layer output to frozen linear layer output
        else:
            lora_weight = self.lora_B @ self.lora_A
            lora_out = F.linear(x, lora_weight) * self.lora_config["alpha"]
            return main_out + lora_out


class FeedForward(nn.Module):
    """
    Simple MLP used in Transformer

    Args:
        dim (int): The input dimensionality
        mult (int): The multiplicative factor, i.e. by which factor to expand
        dropout (float): The amount of dropout used (default 0.1)
    """

    def __init__(self, dim: int, mult: int, dropout: float = 0.1):
        """Initialization"""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x (torch.Tensor): the input tensor (within this project the output of the Transformer block)
        Returns:
            torch.Tensor
        """
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention class. Can be either self- or cross-attention.

    Args
        hidden_dim (int): the last dim of hidden_states tensor
        qkv_dim (int): the dim to run attention on
        context_dim (int): the last dim of the context tensor
        n_heads (int): Number of heads (default: 4)
        dropout (float): Dropout used (default: 0.1)
        lora_config (Dict): pretrain or finetune (default: pretrain -> lora_config = None)

    """

    def __init__(
        self, hidden_dim, qkv_dim, n_heads=4, dropout=0.1, lora_config: Dict = None
    ):
        """Initialization"""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.qkv_dim = qkv_dim
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.lora_config = lora_config

        # mapping from hidden_dim to qkv_dim for q,k and v in one go
        # self.to_q = nn.Linear(self.hidden_dim, qkv_dim * 3, bias=False)
        self.to_qkv = CustomLinear(
            self.hidden_dim, qkv_dim * 3, lora_config=lora_config
        )
        # map back to hidden_dim
        # self.to_out = nn.Linear(qkv_dim, hidden_dim)
        self.to_out = CustomLinear(qkv_dim, self.hidden_dim, lora_config=lora_config)

        # determine dim for each head
        self.head_dim = self.qkv_dim // n_heads
        # scaling factor for multi-head attn
        self.scale = self.head_dim**-0.5
        # init rotary positional encoding
        self.rope = RoPE(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor = None,
        pad_mask: torch.Tensor = None,
        layer_past: Tuple[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor]]:
        """
        Multi-head attention implementation (3d-approach).

        Args:
            hidden_states (torch.Tensor): shape (B, S, M)
            attn_mask (torch.Tensor): shape (S, S)
            pad_mask (torch.Tensor): shape (B, S)
            layer_past (Tuple[torch.Tensor]): containing past key and value tensor

        Returns:
            attention output, attention weights, and updated kv cache
        """
        _, hidden_seq_len, hidden_dim = hidden_states.shape

        # compute query, key and value
        q, k, v = self.to_qkv(hidden_states).chunk(3, dim=-1)

        # note that here q,k and v have the same last dim
        _, _, qkv_dim = q.shape

        # batch-size and n_heads are merged into the first dimension. `d` corresponds
        # to self.head_dim then. Alternatively, heads can be split into a separate dim
        # which is not done here.
        q, k, v = map(
            lambda t: rearrange(t, "b s (d h) -> (b h) s d", h=self.n_heads), (q, k, v)
        )

        # prepare masks such that they can be added via broadcasting
        attn_mask = prepare_attention_mask(attn_mask)
        if pad_mask is not None:
            pad_mask = prepare_padding_mask(pad_mask)
            attn_mask = attn_mask + pad_mask

        # KV-cache handling
        if layer_past is not None:
            # unpack past key and value
            past_key, past_value = layer_past
            # add recent key and value
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        # store updated key and value for later use
        # k and v are 3d tensors of shape (b h), s, d
        present: Tuple[torch.Tensor, ...] = (k, v)

        # Note that during inference, q has a sequence_length of 1 since we only
        # pass the most recent token to the model. However, there is a history/cache,
        # so we need to adjust the rotation angle based on the actual sequence length
        # which is stored in k and v
        _, actual_seq_len, _ = k.shape
        # rotate q according to the actual sequence length
        q = self.rope(
            q, actual_seq_len=actual_seq_len if layer_past is not None else None
        )
        # rotate k
        k = self.rope(k)

        # Compute attention scores using bmm.
        #
        # After the scores have been computed,
        # we directly in one go do the masking by adding a float mask of zeros
        # (non-masked values) and -inf (masked values)

        # The indexing with hidden_seq_len is important for inference since
        # there we might start with less than context_len tokens

        # Special-case if hidden_seq_len is 1, then we are running inference with kv-cache,
        # and we want that this single token can attend to all previous ones. The indexing
        # leaves us with an attn_mask = [[0]] and due to broadcasting we have the expected effect.
        attn_scores = (
            torch.baddbmm(
                attn_mask[:, :hidden_seq_len, :hidden_seq_len], q, k.transpose(-2, -1)
            )
            * self.scale
        )
        # compute weights by applying softmax
        attn_weights: torch.Tensor = self.dropout(F.softmax(attn_scores, dim=-1))

        # computed v according to attention weights
        attn_output: torch.Tensor = torch.bmm(attn_weights, v)

        # rearrange so last dim is qkv dim again
        attn_output = rearrange(attn_output, "(b h) s d -> b s (d h)", h=self.n_heads)
        # project to from qkv_dim to hidden_dim
        attn_output = self.to_out(attn_output)
        # rearrange attention weights to 4d tensor with extra head dimension (not necessary)
        attn_weights = rearrange(attn_weights, "(b h) s d -> b h s d", h=self.n_heads)

        return attn_output, attn_weights, present


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer running multi-head attention multiple times

    Args:
        d_model (int): the input/embedding dimensionality
        n_heads (int): the number of heads used
        mlp_mult_factor (int): the factor by which to expand the input in the FeedForward net (default: 4)
        attn_dropout (float): dropout used in multi-head attention (default: 0.1)
        mlp_dropout (float): dropout used in Feedforward net
        lora_config (Dict): pretrain or finetune (default: pretrain -> lora_config = None)

    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_mult_factor: int = 4,
        attn_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        lora_config: Dict = None,
    ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.ff_norm = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(
            hidden_dim=d_model,
            qkv_dim=d_model,
            n_heads=n_heads,
            dropout=attn_dropout,
            lora_config=lora_config,
        )
        self.ff = FeedForward(dim=d_model, mult=mlp_mult_factor, dropout=mlp_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor = None,
        pad_mask: torch.Tensor = None,
        layer_past: Tuple[torch.Tensor, ...] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor]]:
        """
        Forward pass through decoder layer

        Args:
           hidden_states (torch.Tensor): tensor of shape (B, S, d_model)
           attn_mask (torch.Tensor): tensor of shape (S,S)
           pad_mask (torch.Tensor): tensor of shape (B, S)
           layer_past (Tuple[torch.Tensor]): containing past key and value tensor

        Returns:
            Tuple[torch.Tensor, ..., Tuple[torch.Tensor]]: weighted hidden states, attention weights and updated key-value cache
        """
        # pre-normalize hidden_states and pass through multi-head attention
        y, attn_weights, present = self.mha(
            self.attn_norm(hidden_states), attn_mask, pad_mask, layer_past
        )
        # add hidden_states back to output as residual connection
        out = y + hidden_states
        # pre-normalize output and pass through Feedforward net
        out = self.ff(self.ff_norm(out)) + out  # residual connection

        return out, attn_weights, present


class Transformer(nn.Module):
    """
    Transformer class

    Args:
        d_model (int): the input/embedding dimensionality
        vocab_size (int): the vocabulary size (number of distinct tokens)
        n_layers (int): the number of layers
        n_heads (int): the number of heads
        mlp_mult_factor (int): the factor by which to expand the input in the FeedForward net (default: 4)
        attn_dropout (float): dropout used in multi-head attention (default: 0.1)
        mlp_dropout (float): dropout used in Feedforward net
        lora_config (Dict): pretrain or finetune (default: pretrain -> lora_config = None)

    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        mlp_mult_factor: int = 4,
        attn_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        lora_config: Dict = None,
    ):
        super().__init__()
        # create embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.n_layers = n_layers
        # construct single decoder layer
        self.decoder = TransformerDecoderLayer(
            d_model,
            n_heads,
            mlp_mult_factor,
            attn_dropout,
            mlp_dropout,
            lora_config=lora_config,
        )
        # construct list of decoder layers
        self.layers = _get_clones(self.decoder, self.n_layers)
        # projection layer from d_model to vocab_size
        self.to_logits = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        inp_seq: torch.Tensor,
        attn_mask: torch.Tensor = None,
        pad_mask: torch.Tensor = None,
        past_key_values: Tuple[Tuple[torch.Tensor, ...], ...] = None,
    ) -> Tuple[torch.Tensor, ..., Tuple[Tuple[torch.Tensor, ...], ...]]:
        """
        Forward pass through Transformer

        Args:
            inp_seq (torch.Tensor): tensor  of shape (B, S) containing the raw tokens of a sequence
            attn_mask (torch.Tensor): tensor of shape (S, S)
            pad_mask (torch.Tensor): tensor of shape (B, S)
            past_key_values (Tuple[Tuple[torch.Tensor,...],...]): tuple containing as many kv tuples as there are layers

        Returns:
            prediction tensor of shape (B, S, vocab_size), attention weight tuple, kv-cache tuple
        """
        # for the first run, there are no past_key_values, so create an empty tuple
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        # for each token, lookup its embedding, result is tensor of shape (B, S, d_model)
        word_embedding_tensor = self.embedding(inp_seq)

        out = word_embedding_tensor
        attn_weights: Tuple = ()
        presents: Tuple = ()
        # iterate over every decoder layer block and pass output as new input
        # store intermediate attention weights and kv cache
        for i, (block, layer_past) in enumerate(zip(self.layers, past_key_values)):
            out, attn_weight, present = block(out, attn_mask, pad_mask, layer_past)
            attn_weights = attn_weights + (attn_weight,)
            presents = presents + (present,)

        return self.to_logits(out), attn_weights, presents
