#!python
# -*- coding: utf-8 -*-
"""Probes for observing models."""
__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"

from collections import OrderedDict
from itertools import chain
from typing import TypeVar

import torch
from torch import nn

from utils.model_util import iterate_renderable_layers, build_layers_dict

T = TypeVar('T', bound='ActivationProbe')


class ActivationProbe(nn.Module):
    """
    Model wrapper recording activation as they occur during forward pass.

    Call activation_recording to start recording, then run model forward or
    probe forward method, you will find activations in output_activations,
    input_activations or attentions.
    """

    activation_recording_modes = ["both", "input", "output"]

    def __init__(self, model: nn.Module,
                 verbose: bool = True,
                 activation_recording_mode: str = "both",
                 single_layer_activation_recording: str = None):
        """
        Recursively traverse model layers and attach recording hooks to them.

        :param model: torch.nn.Model to record
        :param activation_recording_mode: ["both", "input", "output"]
        :param single_layer_activation_recording: name of the single layer to record
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
        self.record_activations = False

        for name, layer in self.layers.items():
            layer : nn.Module
            if self.single_layer_activation_recording is None or self.single_layer_activation_recording == name:
                layer.register_forward_hook(self._get_activation_hook(name))

        self.eval()

    def forward(self, *args, return_activations=False, **kwargs):
        """
        Run forward pass on underlying model with activation recording
        :param args:
        :param return_activations: returns model output and all recorded activations
        :param kwargs: kwargs are passed to the model
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
        """Create activation hook to attach to a layer.

        Creates function that parses layer activations and saves them to
        corresponding  dict in self. modify the logic when adding specific layer.
        :param name: name of the layer
        """

        def hook(model, model_input, model_output):
            # if not isinstance(model_output, torch.Tensor):
            #     print("So far so good!")

            if self.record_activations:

                if self.activation_recording_mode == "output" or \
                        self.activation_recording_mode == "both":
                    if isinstance(model, nn.modules.MultiheadAttention):
                        model_output, attention = model_output
                        if attention is None:
                            raise TypeError(f"Attempted to record attentions from "
                                            f"MultiheadAttention layer {name} but "
                                            f"attention is {type(attention)}.\n"
                                            f"In order to record attentions modify "
                                            f"torch.nn.modules.transformer"
                                            f"TransformerDecoderLayer and TransformerEncoderLayer,"
                                            f"in _sa_block and _mha_block, "
                                            f"call self_attn and multihead_attn with need_weights=True.")
                        self.attentions.setdefault(name, []).append(attention.detach())
                        self.output_activations.setdefault(name, []).append(model_output.detach())
                    elif isinstance(model_output, tuple):
                        if self.verbose: print(f"Skip tuple {name}")
                    elif isinstance(model_output, list):
                        self.output_activations[name] = [o.detach() for o in model_output]
                    elif isinstance(model_output, dict):
                        if self.verbose: print(f"Skip dict {name}")
                    else:
                        self.output_activations[name] = model_output.detach()

                if self.activation_recording_mode == "input" or \
                        self.activation_recording_mode == "both":
                    self.input_activations[name] = model_input[0].detach()
        return hook
