"""Memory controls that retain full attention context and target supervision."""
from types import MethodType

import torch
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP


def chunk_text_mlps(model, chunk_size: int) -> int:
    """Checkpoint token-wise MLP chunks; preserve weights, keys and token order.

    Evaluation uses the same chunks without autograd checkpointing. Chunking
    changes dropout RNG ordering versus an unchunked run, so its size belongs
    in the reproducibility contract. Attention is never split into windows.
    """
    if chunk_size < 0:
        raise ValueError("MLP chunk size must be nonnegative")
    if not chunk_size:
        return 0

    def wrap(original):
        def forward(self, hidden_states):
            def compute(states):
                return original(self, states)

            chunks = [checkpoint(compute, states, use_reentrant=False)
                      if self.training and torch.is_grad_enabled() else compute(states)
                      for states in hidden_states.split(chunk_size, dim=-2)]
            return torch.cat(chunks, dim=-2)
        return forward

    count = 0
    for name, module in model.named_modules():
        if ".language_model." not in name or not isinstance(module, Qwen3_5MLP):
            continue
        previous = getattr(module, "_finetune_mlp_chunk_size", None)
        if previous is not None:
            if previous != chunk_size:
                raise ValueError("Cannot change MLP chunk size on an already configured model")
        else:
            module.forward = MethodType(wrap(module.forward.__func__), module)
            module._finetune_mlp_chunk_size = chunk_size
        count += 1
    if not count:
        raise ValueError("No Qwen3.5 text MLP modules found")
    return count
