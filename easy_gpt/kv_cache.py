import torch
from typing import Tuple


class KVUpdater:
    """
    Key-value cache updating according to Attention sink idea
    see: https://github.com/mit-han-lab/streaming-llm

    Args:
        keep_n_recent: the window size for inference (how much past would you like to consider for prediction?)

    """

    def __init__(self, keep_n_recent: int):
        super().__init__()
        # the context len
        # we always have to reserve a spot for the first token
        self.keep_n_recent = keep_n_recent
        self.cache_size = keep_n_recent + 1  # special attention sink token

    def throw_away(
        self, past_key_values: Tuple[Tuple[torch.Tensor, ...], ...], num_coming: int
    ) -> Tuple[Tuple[torch.Tensor, ...], ...]:
        """
        Update logic for kv-cache. If the amount of tokens that are about to be predicted
        go beyond the window-size (keep_n_recent), we have to throw away from the beginning of
        the kv-cache BUT always keep the first token (attention sink).

        Args:
            past_key_values: tuple containing kv-tuples of each decoder layer
            num_coming: the number of tokens to be predicted

        Returns:
            the updated tuple of kv-tuples of each decoder layer
        """
        if past_key_values is None:
            return None

        seq_len = past_key_values[0][0].size(2)
        # if we are about to generate beyond the context len,
        # we have to throw away parts of the kv-cache
        # BUT: Always keep attention sink token (the very first)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values

        # an example: say we have generated 190 tokens, our cache size
        # is 201, but we want to generate 20 new tokens. We need to have
        # 201-20 = 181 kv-cache such that the 20 new ones fit in.
        # Since we have only 11 left but need 20, we have to throw
        # away the first 10 tokens and start selecting from 10-190
        start = seq_len - self.keep_n_recent + num_coming
        return tuple(
            [
                (
                    torch.cat([k[:, 0:1, :], k[:, start:seq_len, :]], dim=1),
                    torch.cat([v[:, 0:1, :], v[:, start:seq_len, :]], dim=1),
                )
                for k, v in past_key_values
            ]
        )
