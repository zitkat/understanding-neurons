#!python
# -*- coding: utf-8 -*-
"""
Utilities for manipulating model layers.
"""

__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"

from pathlib import Path

from typing import Dict, List, OrderedDict

import torch
import timm
from lucent.optvis import objectives
from lucent.optvis.objectives import Objective
from torch import nn as nn


from .process_util import plogger, now, ensured_path


def build_layers_dict(module : nn.Module):
    """
    Like get_model_layers from lucent.modelzoo.util
    but returns actual layer objects
    :param module:
    :return: OrderedDict(layer_name: layer)
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


def get_layer(module : nn.Module, layer : str):
    """
    Traverses module base on dash - connectd
    string to extract layer object
    """
    last_layer = module
    layer_path = layer.split("-")
    for step in layer_path:
        if step.isdigit():
            last_layer = last_layer[int(step)]
        else:
            last_layer = getattr(last_layer, step)
    return last_layer


def is_composite(layer : nn.Module) -> bool:
    """
    Returns True if layer contains any modules
    """
    return bool(getattr(layer, "_modules", ()))


def count_renderable_units(layer : nn.Module):
    """Return number of renderable units in layer,
        so far support only Conv and Linear
    """
    return (getattr(layer, "out_channels", 0)    # Conv
            # + getattr(layer, "out_features", 0)    # Linear
            )


def iterate_renderable_layers(layers : Dict[str, nn.Module],  verbose=False):
    for layer_name, layer_object in layers.items():
        n = count_renderable_units(layer_object)
        if n > 0:
            if verbose:
                plogger.debug(f"\n\n{now()} Starting layer {layer_name}\n")
            yield layer_name, layer_object, n
        else:
            if verbose:
                plogger.debug(f"{now()} Skipping layer {layer_name}")
            continue


def ncobj(m : str, layer : str, n : int, batch=0) -> Objective:
    """Constructs objective based on type m for layer and unit within the layer"""
    obj_constructor = getattr(objectives, m)
    return obj_constructor(layer.replace("-", "_"), n , batch=batch)


def batch_indices(indcs : List, batch_size : int):
    indcs = list(indcs)
    start = 0
    end = batch_size
    while start < len(indcs):
        yield indcs[start:end]
        start = end
        end += batch_size


def get_timm_classfier(architecture_name : str, target_size : int, pretrained : bool = False) -> nn.Module:
    """
    Gets model from timm with new classifier with target_size classes
    :param architecture_name: see list of models available from timm
    :param target_size: number of classes of the new classifier
    :param pretrained: get pretrained weights for feature extractor
    :return: new model classifier with target_size output classes
    """
    net = timm.create_model(architecture_name, pretrained=pretrained)
    net_cfg = net.default_cfg
    last_layer = net_cfg['classifier']
    num_ftrs = getattr(net, last_layer).in_features
    setattr(net, last_layer, nn.Linear(num_ftrs, target_size))
    return net


def get_model(model_name, model_weights, output):
    """
    Get model from name and weight specification.
    :param model_name: model name in timm
    :param model_weights: path to weights or pretrained to
            download pretrained wights from timm, or initialized
            to use rendomly initialized
    :param output: output folder to save initial weights for reproducibility
    :return: model, weights name
    """
    name = model_weights
    if model_weights.endswith(".pth"):
        model_weights_path = Path(model_weights)
        name = model_weights_path.stem
        model = get_timm_classfier(model_name, target_size=5)
        net_dict = torch.load(model_weights)
        model.load_state_dict(net_dict)
    elif model_weights == "pretrained":
        plogger.info(f"Loading pretrained model from timm.")
        model = timm.create_model(model_name, pretrained=True)
    elif model_weights == "initialized":
        save_path = ensured_path(output / (model_name + "_init.pth"))
        if save_path.exists():
            plogger.info(f"Loading existing initialization from {save_path}")
            model = get_timm_classfier(model_name, target_size=5)
            net_dict = torch.load(save_path)
            model.load_state_dict(net_dict)
        else:
            plogger.info(f"Saving new initialization to {save_path}")
            model = get_timm_classfier(model_name, target_size=5, pretrained=False)
            torch.save(model.cpu().state_dict(), save_path)
    else:
        err_msg = f"Unknown option for model weights {model_weights} terminating!"
        plogger.error(err_msg)
        raise ValueError(err_msg)

    return model, name,

