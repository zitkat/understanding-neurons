#!python
# -*- coding: utf-8 -*-

__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"
from pathlib import Path

import time

import numpy as np
import timm
from lucent.optvis import objectives
from torch import nn as nn


def make_path(*pathargs, isdir=False, **pathkwargs):
    new_path = Path(*pathargs, **pathkwargs)
    return ensured_path(new_path, isdir=isdir)


def ensured_path(path: Path, isdir=False):
    if isdir:
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def now():
    """
    :return: date and time as YYYY-mm-dd-hh-MM
    """
    return time.strftime("%Y-%m-%d-%H-%M")


def get_layer(module : nn.Module, layer : str):
    """
    Traverses module base on underscore _ connectd
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


def is_composite(layer):
    return bool(getattr(layer, "_modules", ()))


def renderable_units(layer : nn.Module):
    """Return number of renderable units in layer,
        so far support only Conv and Linear
    """
    return getattr(layer, "out_channels", 0) + getattr(layer, "out_features", 0)



def ncobj(t : str, layer : str, n : int, batch=0):
    """Constructs objective base on type t"""
    obj_constructor = getattr(objectives, t)
    return obj_constructor(layer.replace("-", "_"), n , batch=batch)


def batch_indices(indcs, batch_size):
    indcs = list(indcs)
    start = 0
    end = batch_size
    while start < len(indcs):
        yield indcs[start:end]
        start = end
        end += batch_size


def get_timm_model(architecture_name, target_size, pretrained=False):
    net = timm.create_model(architecture_name, pretrained=pretrained)
    net_cfg = net.default_cfg
    last_layer = net_cfg['classifier']
    num_ftrs = getattr(net, last_layer).in_features
    setattr(net, last_layer, nn.Linear(num_ftrs, target_size))
    return net


def get_unit_from_labels(labels: np.ndarray, layer_name, n, prefix="neurons", suffix="v1"):
    wherelay = np.where(labels == "-".join((prefix, layer_name, suffix)))[0]
    lidx = wherelay[0]
    if n > len(wherelay):
        raise ValueError(f"Only {len(wherelay)} in layer {layer_name} rendered but {n}-th one requested")
    nidx = lidx + n
    return nidx


def pretty_layer_label(layer_name, n=None):
    """Pretty prints layer name"""
    parts = layer_name.split("-")
    if len(parts) <= 3:
        return parts[1]
    return f"L{parts[1][-1]}.{parts[2]} Conv {parts[3][-1]} : {none2str(n)}"


def get_var_filter_iter(df, var=None):
    """
    Returns number of unique values of var in df and iterator going over masks
    selecting for each value. If var is None only one full mask is yielded.
    :return: len(vals), Yield[val, df[var] == val]
    """
    if var is None:
        vals = [""]
    else:
        vals = df[var].unique()

    def val_iterator():
        if var is None:
            yield "", np.ones(len(df), dtype=bool)
        else:
            for val in vals:
                yield val, df[var] == val

    return len(vals), val_iterator


def fill_dict(keys, vals):
    """
    Fills dictionary with not None values.
    """
    return {key: val for key, val in zip(keys, vals) if val is not None}


def none2str(v):
    """
    str() wrapper, if v is None returns empty string.
    """
    if v is None:
        return ""
    return str(v)


def split_seresnext_labels(df):
    df["label"] = df["label"].str.split("_").apply(lambda s: "_".join(s[1:]))
    df["bottleneck"] = df["label"].str.split("_").apply(lambda s: s[-3][-1] if len(s) >= 3 else "-")
    df["layer"] = df["label"].str.split("_").apply(lambda s: s[0])
    df["conv"] = df["label"].str.split("_").apply(lambda s: s[-2][-1])
    return df


def load_npy_fvs(input_path: Path, mode="neurons", version="v1"):
    data_paths = list(input_path.glob(f"{mode}*{version}*.npy"))
    data_list = [np.load(dpath) for dpath in data_paths]
    data_labels = sum(([dpth.name[:-4],] * len(d)
                       for dpth, d in zip(data_paths, data_list)),
                      start=[])
    data = np.concatenate(data_list, axis=0)
    data_labels = np.array(data_labels)
    return data, data_labels