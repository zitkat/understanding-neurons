#!python
# -*- coding: utf-8 -*-

__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"

import json
from pathlib import Path

import numpy as np
import pandas as pd

def get_unit_from_labels(labels: np.ndarray, layer_name, n, prefix="neurons", suffix="v1"):
    wherelay = np.where(labels == "-".join((prefix, layer_name, suffix)))[0]
    lidx = wherelay[0]
    if n > len(wherelay):
        raise ValueError(f"Only {len(wherelay)} in layer {layer_name} rendered but {n}-th one requested")
    nidx = lidx + n
    return nidx


def pretty_layer_label(layer_name, n=None, sep="-"):
    """Pretty prints layer name"""
    parts = layer_name.split(sep)
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


def split_mobilenet_labels(df : pd.DataFrame, sep="-"):
    df["label"] = df["label"].str.split(sep).apply(lambda s: sep.join(s[1:-1]))
    df["se"] = df["label"].str.contains("se")
    df["block"] = df["label"].str.split(sep).apply(lambda s: s[1] if len(s) >= 3 else s[-1])
    df["branch"] = df["label"].str.split(sep).apply(lambda s: s[2] if len(s) >= 3 else "0")
    df["conv"] = df["label"].str.split(sep).apply(lambda s: s[-2] if len(s) >= 3 else s[-1])
    df["new_index"] = df["label"] + ":" + df["unit"].map(str)

    return df.set_index("new_index")


def load_npy_fvs(input_path: Path, mode="neurons", version="v1"):
    data_paths = list(input_path.glob(f"{mode}*{version}*.npy"))
    data_list = [np.load(dpath, allow_pickle=True) for dpath in data_paths]
    data_labels = sum(([dpth.name[:-4],] * len(d)
                       for dpth, d in zip(data_paths, data_list)),
                      start=[])
    data_units = [np.arange(0, len(dat)) for dat in data_list]

    data = np.concatenate(data_list, axis=0)
    data_labels = np.array(data_labels)
    data_units = np.concatenate(data_units, axis=0)
    return data, data_labels, data_units


def load_criticalstats_dict(input_path : Path):
    with open(input_path, "r", encoding="utf-8") as f:
        rawjson = json.load(f)
    return rawjson


def add_criticality_data(p_df, from_json):
    c_dict = load_criticalstats_dict(from_json)
    new_order = []
    for ii, (crit_col, crits_list) in enumerate(c_dict.items()):
        p_df[crit_col] = 0
        for jj, layer_dict in enumerate(crits_list):
            for layer_name, units_crits in layer_dict.items():
                print(f"Layer {layer_name}: {len(p_df[p_df.label == layer_name])} ~ {len(units_crits)}")
                for unit, crits in units_crits.items():
                    crits = np.array(list(map(float, crits)))
                    p_df.loc[f"{layer_name}:{unit}", crit_col] = np.mean(crits)
                    new_order.append(f"{layer_name}:{unit}")

    return p_df.reindex(new_order)
