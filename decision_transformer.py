from model import GPTConfig, GPT
from torch import nn

class DecisionTransformer(nn.Module):
    """
    Reimplementation of the Decision Transformer with latest optimizations and speedups.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        