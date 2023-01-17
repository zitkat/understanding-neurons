#!python
# -*- coding: utf-8 -*-
"""Probes for observing models."""
__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"

from collections import OrderedDict
from itertools import chain
from typing import TypeVar

import toolz

import torch
from torch import nn

from deformable_potr.models.ops.modules import MSDeformAttn
from .utils.model_util import iterate_renderable_layers, build_layers_dict


T = TypeVar('T', bound='ActivationProbe')


class ActivationProbe(nn.Module):
    """
    Model wrapper providing usefull functionality for analyzing a module.
    """

    activation_recording_modes = ["both", "input", "output"]

    def __init__(self, model: nn.Module,
                 verbose: bool = True,
                 activation_recording_mode: str = "both",
                 single_layer_activation_recording: str = None):
        """

        :param model: torch.nn.Model to map
        :param activation_recording_mode: ["both", "input", "output"]
        """

        super(ActivationProbe, self).__init__()
        self.verbose = verbose
        self.module = model
        self.layers: OrderedDict[torch.nn.Module] = build_layers_dict(self.module)

        self.renderable_layers: OrderedDict[torch.nn.Module] = \
            OrderedDict((n, o) for n, o, _ in iterate_renderable_layers(self.layers))

        self.activation_recording_mode = "both"
        self.change_activation_rec_mode(activation_recording_mode)

        if single_layer_activation_recording is not None:
            if single_layer_activation_recording not in self.layers:
                raise ValueError(f"Layer {single_layer_activation_recording} not present in model layers")

        self.single_layer_activation_recording = single_layer_activation_recording

        self.output_activations: OrderedDict[torch.Tensor] = OrderedDict()
        self.input_activations: OrderedDict[torch.Tensor] = OrderedDict()
        self.attentions: OrderedDict[torch.Tensor] = OrderedDict()
        self.sampling_locations: OrderedDict[torch.Tensor] = OrderedDict()
        self.record_activations = False

        for name, layer in self.layers.items():
            layer : nn.Module
            if self.single_layer_activation_recording is None or self.single_layer_activation_recording == name:
                layer.register_forward_hook(self._get_activation_hook(name))

        self.eval()

    def forward(self, *args, return_activations=False, **kwargs):
        """
        Forward pass on underlying model with activation recordning
        :param args:
        :param return_activations:
        :param kwargs:
        :return:
        """
        out = self.module.forward(*args, **kwargs)
        if self.record_activations and return_activations:
            if self.activation_recording_mode == "both" or \
                    self.activation_recording_mode == "output":
                return self.output_activations
            else:
                return OrderedDict(chain(self.input_activations.items(),  OrderedDict(out=out).items()))
        return out

    def __call__(self, *args, return_activations=False, **kwargs):
        return self.forward(*args, return_activations=return_activations, **kwargs)

    def train(self: T, mode: bool = True) -> T:
        return super(ActivationProbe, self).train(mode)

    def activation_recording(self: T, mode : bool) -> T:
        self.record_activations = mode
        return self

    def pause_activation_rec(self: T) -> T:
        self.record_activations = False
        return self

    def resume_activation_rec(self : T) -> T:
        self.record_activations = True
        return self

    def change_activation_rec_mode(self : T, newmode : str) -> T:
        if newmode in ActivationProbe.activation_recording_modes:
            self.activation_recording_mode = newmode
            return self
        else:
            raise ValueError("Unknown activation recording mode.")

    def clear_records(self: T) -> T:
        self.input_activations = OrderedDict()
        self.output_activations = OrderedDict()
        self.attentions = OrderedDict()
        return self

    def silence(self: T) -> T:
        self.verbose = False
        return self

    def verbose(self: T, val: bool) -> T:
        self.verbose = val
        return self

    def _get_activation_hook(self: T, name):

        def hook(model, model_input, model_output):
            # if not isinstance(model_output, torch.Tensor):
            #     print("So far so good!")

            if self.record_activations:

                if self.activation_recording_mode == "output" or \
                        self.activation_recording_mode == "both":
                    if isinstance(model, nn.modules.MultiheadAttention):
                        model_output, attention = model_output
                        self.attentions.setdefault(name, []).append(attention.detach())
                    elif isinstance(model, MSDeformAttn):
                        model_outpout, sampling_locations, attention_weights = model_output
                        self.attentions.setdefault(name, []).append(attention_weights.detach())
                        self.sampling_locations.setdefault(name, []).append(sampling_locations.detach())
                    elif isinstance(model_output, tuple):
                        if self.verbose: print(f"Skip tuple {name}")
                    elif isinstance(model_output, list):
                        self.output_activations[name] = [o.detach() for o in model_output]
                    elif isinstance(model_output, dict):
                        # self.output_activations[name] = toolz.valmap(toolz.partial(map, torch.detach), model_output)
                        if self.verbose: print(f"Skip dict {name}")
                    else:
                        self.output_activations[name] = model_output.detach()

                if self.activation_recording_mode == "input" or \
                        self.activation_recording_mode == "both":
                    self.input_activations[name] = model_input[0].detach()
        return hook