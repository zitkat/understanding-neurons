#!python
# -*- coding: utf-8 -*-
"""
Model analysis tools wrapped in MappedModel class.
"""
__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"

from collections import OrderedDict
from itertools import chain
from typing import TypeVar
from pathlib import Path

import torch
from lucent.optvis import render
from torch import nn

import multi_renders
from utils.model_util import iterate_renderable_layers, build_layers_dict


T = TypeVar('T', bound='MappedModel')


class MappedModel(nn.Module):
    """
    Model wrapper providing usefull functionality for analyzing a module.
    """

    activation_recording_modes = ["both", "input", "output"]

    def __init__(self, model: nn.Module,
                 activation_recording_mode : str = "both",
                 single_layer_activation_recording : str = None):
        """

        :param model: torch.nn.Model to map
        :param activation_recording_mode: ["both", "input", "output"]
        """

        super(MappedModel, self).__init__()
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
        return super(MappedModel, self).train(mode)

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
        if newmode in MappedModel.activation_recording_modes:
            self.activation_recording_mode = newmode
            return self
        else:
            raise ValueError("Unknown activation recording mode.")

    def clear_records(self: T) -> T:
        self.input_activations = OrderedDict()
        self.output_activations = OrderedDict()
        self.attentions = OrderedDict()
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
    mmodel = MappedModel(model, activation_recording_mode="input").eval().to(device)
    act = mmodel.forward(torch.zeros((1, 3, 224, 224)).to(device), return_activations=True)
    print(model)

    all_layers = list(mmodel.layers.keys())
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
