#!python
# -*- coding: utf-8 -*-
"""Probes for observing models."""
__author__ = "Tomas Zitka"
__email__ = "tozitka@gmail.com"

from collections import OrderedDict
from itertools import chain
from typing import TypeVar
from pathlib import Path

import torch
from lucent.optvis import render
from torch import nn

from visualization import multi_renders
from utils.pytorch_model_util import iterate_renderable_layers, build_layers_dict

T = TypeVar('T', bound='ActivationProbe')


class ActivationProbe(nn.Module):
    """
    PyTorch Model wrapper recording activations as they occur during forward pass.

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

    def extract_circuit(self : T, layer, n, extraction_strategy=None):
        head_weights = self[layer, n]
        # TODO use https://pytorch.org/docs/stable/jit.html to get traversable computational graph?

    def render_vis(self : T, *args, **kwargs):
        return render.render_vis(self.module, *args, **kwargs)

    def render_layer(self : T, *args, **kwargs):
        return multi_renders.render_layer(self.module, *args, **kwargs)

    def render_model(self : T, *args, **kwargs):
        return multi_renders.render_model(self.module, *args, **kwargs)

    def __getitem__(self : T, item):
        if isinstance(item, slice):
            # TODO return slice of layers as invocable, mapped module
            raise NotImplemented("TODO return slice of layers as invocable, mapped module")
        elif isinstance(item, list):
            # TODO return list of layers
            raise NotImplemented("TODO reuturn list of layers")
        elif isinstance(item, tuple):
            return self.layers[item[0]].weight[int(item[1])]
        elif isinstance(item, str):
            if ":" in item:
                pref, suf = item.split(":")
                return self.layers[pref].weight[int(suf)]
            return self.layers[item]


if __name__ == '__main__':
    import timm

    model = timm.create_model("resnet50", pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pmodel = ActivationProbe(model, activation_recording_mode="input").eval().to(device)
    act = pmodel.forward(torch.zeros((1, 3, 224, 224)).to(device), return_activations=True)
    print(model)

    all_layers = list(pmodel.layers.keys())
    rendered_path = Path("data/pretrained_seresnext50_32x4d/npys")
    rendered_layers = list(rendered_path.glob("*.npy"))
    all_conv_layers = list(filter(lambda s: "conv" in s, all_layers))
    rendered_layers = ["_".join(fl.stem.split("_")[1:-1]) for fl in rendered_layers]
    print("All layers", len(all_conv_layers))
    len(rendered_layers)
    print("Rendered layers", len(set(rendered_layers)))
    todo_layers = list(set(all_conv_layers) - set(rendered_layers))
    print("TODO layers", len(todo_layers))
    open(rendered_path.parent / "layers.list", "w").write("\n".join(todo_layers))
