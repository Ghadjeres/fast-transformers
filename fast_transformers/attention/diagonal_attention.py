#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement diagonal attention"""

import torch
from torch.nn import Module

from ..attention_registry import AttentionRegistry, Optional, Callable, \
    EventDispatcherInstance
from ..events import EventDispatcher


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class DiagonalAttention(Module):
    """Simply returns values

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
        super(DiagonalAttention, self).__init__()
        self.feature_map = feature_map or elu_feature_map
        self.eps = eps
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)


    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        return values


# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "diagonal", DiagonalAttention,
    [
        ("feature_map", Optional(Callable)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
