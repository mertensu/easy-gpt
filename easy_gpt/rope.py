import torch
import torch.nn as nn
from einops import rearrange, repeat


class RoPE(nn.Module):
    """
    Rotary positional encoding as introduced here: https://arxiv.org/abs/2104.09864
    Adjusted (simplified) implementation of https://github.com/lucidrains/rotary-embedding-torch

    Args:
        dim (int): the dimensionality of the input (d_model)

    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # compute frequencies for each block of 2
        self.freqs = nn.Parameter(
            torch.tensor(
                [self.compute_freq(pos, self.dim) for pos in range(0, self.dim, 2)],
            ),
            requires_grad=False,
        )
        # construct placeholder for position in the sequence
        self.pos_index = nn.Parameter(
            torch.arange(start=0.0, end=10_000), requires_grad=False
        )

    @staticmethod
    def compute_freq(pos: int, dim: int) -> float:
        """
        Return rotation amount for a pair along the input dimensionality.

        Args:
            pos: the respective position of the pair
            dim: the total dimensionality

        Returns:

        """
        return 1.0 / (10_000 ** (pos / dim))

    def compute_rotation_amount(self, seq_len: int) -> torch.Tensor:
        """
        Compute a tensor holding for each value the corresponding rotation amount.

        The frequencies (rows) are multiplied with the respective positions in the sequence
        freq = [1,3,5], pos = [0,1,2,3,4]
        result
        [[0,0,0] -> freq * 0
        [1,3,5]  -> freq * 1
        [2,6,10] -> freq * 2
        [3,9,15] ...
        [4,12,20]] ...

        Args:
            seq_len: the sequence length

        Returns:
            torch.Tensor: The result tensor of shape (seq_len, dim)
        """
        # get position index
        pos_index = self.pos_index[:seq_len]
        # repeat every element to be on original dim again
        # remember above we compute the rotation only for every pair,
        # but we want to repeat now (1,2,3) becomes (1,1,2,2,3,3)
        freqs = repeat(self.freqs, "... n -> ... (n r)", r=2)
        # return tensor of shape seq_len, dim
        return torch.matmul(pos_index[:, None], freqs[None, :])

    @staticmethod
    def switch_blocks(q_or_k: torch.Tensor) -> torch.Tensor:
        """
        For the rotation matrix, we need to compute sin on the first element and -sin on the second element.
        We can use a little trick, namely switching the order of every block such that an order of:

        0,1,2,3,4,5,6,7 is changed to 1,0,3,2,5,4,7,6. If we also negate each first element in the blocks, so:

        -1,0,-3,2,-5,4,-7,6 we can just apply the sin and get the desired result
        -> +sin(0,2,4,6) and -sin(1,3,5,7)

        Args:
            q_or_k (torch.Tensor): either query or key of shape (seq_len, dim)

        Returns:

        """
        # let's assume d = 10
        # make a small sub-matrix of shape 5, 2
        x = rearrange(q_or_k, "... (d r) -> ... d r", r=2)
        # split last dimension into two sub-tensors each of shape 5
        x1, x2 = x.unbind(dim=-1)
        # stack them together in reverse order and negate second elements,
        # so we get back a tensor of shape 5, 2
        x = torch.stack((-x2, x1), dim=-1)
        # back to original shape d=10
        return rearrange(x, "... d r -> ... (d r)")

    def forward(self, q_or_k: torch.Tensor, actual_seq_len: int = None) -> torch.Tensor:
        """
        Apply the rotation of query or key. When using kv-caching, we pass only the last token
        to the Transformer and not the sequence of tokens generated so far. Reason: We have the
        information already stored in the kv-cache. But this means that sequence length would be 1.
        However, this is not true regarding rotation, so we should not look at the shape of the passed
        tensor (q_or_k) but rather pass the actual sequence length from the kv-cache

        Args:
            q_or_k: query or key of shape (seq_len, dim)
            actual_seq_len: the sequence length from the kv-cache

        Returns:
            torch.Tensor: The rotated query or key
        """
        assert len(q_or_k.shape) == 3, "assumes tensor with bs, seq_len, d"
        # unpack shape
        bs, seq_len, d = q_or_k.shape
        # simple implementation where every value along the row has to be rotated
        assert self.dim == d, "dim should match last dimension of provided tensor"
        if actual_seq_len is not None:
            # used during inference for q when kv-cache is used
            freqs = self.compute_rotation_amount(actual_seq_len)[-1, :]
        else:
            freqs = self.compute_rotation_amount(seq_len)

        # apply rotation matrix
        return q_or_k * freqs.cos() + self.switch_blocks(q_or_k) * freqs.sin()


if __name__ == "__main__":
    rope = RoPE(dim=8)

    q = torch.ones(1, 13, 8)
    _, seq_len, _ = q.shape
    rot_q = rope(q)

    print(q)
    print(rot_q)
    print(rope.compute_rotation_amount(seq_len).shape)
