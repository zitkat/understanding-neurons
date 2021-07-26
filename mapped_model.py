#!python
# -*- coding: utf-8 -*-

__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"

from collections import OrderedDict
from typing import TypeVar
from pathlib import Path
import os
import json

import torch
from lucent.optvis import render
from torch import nn

import multi_renders
from utils.model_util import iterate_renderable_layers
from utils.dataset_util import DataSet
from utils.safety_util import SafetyAnalysis
from utils.vis_util import plot_cdp_results


T = TypeVar('T', bound='MappedModel')


class MappedModel(nn.Module):

    def __init__(self, model):
        super(MappedModel, self).__init__()
        self.module = model
        self.layers = build_layers_dict(self.module)

        self.renderable_layers = OrderedDict(
                (n, o) for n, o, _ in iterate_renderable_layers(self.layers))

        self.activations = OrderedDict()
        self.record_activations = False
        for name, layer in self.layers.items():
            layer : nn.Module
            layer.register_forward_hook(self._get_activation_hook(name))


    def forward(self, *args, return_activations=False, **kwargs):
        out = self.module.forward(*args, **kwargs)
        if self.record_activations and return_activations:
            return self.activations
        return out

    def train(self: T, mode: bool = True) -> T:
        self.record_activations = not mode
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


    def _get_activation_hook(self, name):
        def hook(model, input, output):
            if self.record_activations:
                self.activations[name] = output.detach()
        return hook


    def extract_circuit(self, layer, n, extraction_strategy=None):
        head_weights = self[layer, n]
        # TODO use https://pytorch.org/docs/stable/jit.html to get traversable computational graph?

    def kill_unit(self, layer, unit) -> nn.Module:
        ...
        # TODO zero out specified unit

    def render_vis(self, *args, **kwargs):
        return render.render_vis(self.module, *args, **kwargs)

    def render_layer(self, *args, **kwargs):
        return multi_renders.render_layer(self.module, *args, **kwargs)

    def render_model(self, *args, **kwargs):
        return multi_renders.render_model(self.module, *args, **kwargs)

    def __getitem__(self, item):
        if isinstance(item, slice):
            # TODO return slice of layers as invocable, mapped module
            raise NotImplemented("TODO return slice of layers as invocable, mapped module")
        elif isinstance(item, list):
            # TODO reuturn list of layers
            raise NotImplemented("TODO reuturn list of layers")
        elif isinstance(item, tuple):
            # TODO return neuron weights based on (layer, n) tuple
            raise NotImplemented("TODO return neuron weights base on (layer, n) tuple")
        elif isinstance(item, str):
            if ":" in item:
                pref, suf = item.split(":")
                return self.layers[pref].weight[int(suf)]
            return self.layers[item]


def build_layers_dict(module : nn.Module):
    """
    Like get_model_layers from lucent.modelzoo.util
    but returns actual layer objects
    :param module:
    :return: OrderedDict(layer_name: layer
    """
    layers = OrderedDict()

    def get_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    continue
                layers["-".join(prefix+[name])] = layer
                get_layers(layer, prefix=prefix+[name])

    get_layers(module)
    return layers


if __name__ == '__main__':
    import timm

    model = timm.create_model("mobilenetv3_rw", pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    mmodel = MappedModel(model).eval().to(device)
    print(model)

    with open(os.path.join("data", "statistics_dict.json")) as f:
        statistics = json.load(f)
        plot_cdp_results(os.path.join("data", "generated"), statistics, "mobilenetv3_rw", 0.5)

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