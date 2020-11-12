import torch
from torch.nn import Module

from ....attention_registry import RecurrentCrossAttentionRegistry, Optional, \
    Callable, EventDispatcherInstance
from ....events import EventDispatcher



class RecurrentDiagonalAttention(Module):
    """Implement autoregressive linear cross attention as a recurrent
    module.

    See fast_transformers.attention.linear_attention.LinearAttention .

    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, feature_map=None, eps=1e-6, event_dispatcher=""):
        super(RecurrentDiagonalAttention, self).__init__()
        self.feature_map = None
        self.eps = eps
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, query, keys, values, key_lengths, state=None):
        if state is None:
            state = 0
        else:
            state += 1
        value = values[:, state]
        return value, state


# Register the attention implementation so that it becomes available in our
# builders
RecurrentCrossAttentionRegistry.register(
    "diagonal", RecurrentDiagonalAttention,
    [
        ("feature_map", Optional(Callable)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
