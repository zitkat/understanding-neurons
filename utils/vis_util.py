#!python
# -*- coding: utf-8 -*-

__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import collections
import os
import itertools

import pandas as pd


def get_unit_from_labels(labels: np.ndarray, layer_name, n, prefix="neurons", suffix="v1", sep="-"):
    wherelay = np.where(labels == sep.join((prefix, layer_name, suffix)))[0]
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


def plot_cdp_results(_path,
                     _cri_stat_dict,
                     _model_name,
                     _cri_tau=0.5):

    top_x_neurons = 100
    layers_name = None

    fig, ax = plt.subplots()

    for label, labels_data in _cri_stat_dict.items():

        layers_criticality = collections.defaultdict()

        fig_dir = os.path.join(_path, _model_name, label, "layers_criticality")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        for layers_data in tqdm(labels_data):

            criticality_values = list()
            kernel_indices = list()

            for layers_names, kernels_criticalities in layers_data.items():
                layers_name = layers_names
                for kernels_index, kernel_criticality in kernels_criticalities.items():
                    kernel_indices.append(int(kernels_index))
                    criticality_values.append(np.mean([float(value) for value in kernel_criticality]))
                    # criticality_values.append( np.mean(only_critical_neurons) * (max_number_of_weights / len(top_x_values)) )

                if top_x_neurons == "all":
                    top_x_values = criticality_values
                    top_x_indices = kernel_indices
                    plt.rcParams.update({'font.size': 8})
                else:
                    max_index = min(top_x_neurons, len(criticality_values))
                    indices = np.argsort(criticality_values)
                    top_x_indices = indices[-max_index:]
                    top_x_values = [criticality_values[index] for index in top_x_indices]
                    plt.rcParams.update({'font.size': 6})

                # criticality pro layer
                only_critical_neurons = [criticality_value
                                         for criticality_value in top_x_values if
                                         criticality_value > _cri_tau]

                colors = list()
                for values in top_x_values:
                    if values > _cri_tau:
                        colors.append('red')
                    elif values > _cri_tau / 2:
                        colors.append('orange')
                    elif values > _cri_tau / 10:
                        colors.append('yellow')
                    elif values < 0.0:
                        colors.append('blue')
                    else:
                        colors.append('green')

                ''' --------------------------------------- '''
                ''' Plotting histogram of L2-norms '''
                ''' --------------------------------------- '''
                # clear the previous axis
                ax.clear()
                ax.barh(np.arange(len(top_x_values)), top_x_values, alpha=0.5, color=colors, align='center')
                ax.invert_yaxis()  # labels read top-to-bottom
                ax.set_yticklabels(top_x_indices, minor=False)
                ax.set_xlabel('Criticality')
                ax.set_ylabel('Indices')
                #ax.set_title("Neural criticality for layer: {} and class: {}".format(each_layer, _class))

                # fig.suptitle("Neurons\' criticality for class : pedestrian")
                fig.tight_layout()
                fig.savefig(os.path.join(fig_dir, layers_name + "_CNA_result.png"))

                ''' --------------------------------------- '''
                ''' Plotting histogram of L2-norms '''
                ''' --------------------------------------- '''
                # clear the previous axis
                ax.clear()
                bin = 100
                n, bins, patches = ax.hist(top_x_values, bin, density=True, facecolor='g', alpha=0.75)

                mean = np.mean(top_x_values)
                std = np.std(top_x_values)
                entropy = False
                if entropy:
                    newX_top_x_values = top_x_values - mean
                    newX_top_x_values = newX_top_x_values / std
                    layers_entropy = entropy(newX_top_x_values, base=2)
                    ax.text(0, .5, layers_entropy)

                # ax.set_xlabel('Filter indexes')
                ax.set_ylabel("Density")
                #ax.set_title("Histogram of normalized criticality for layer: {} and class: {}".format(each_layer, _class))

                # ax.grid(True)
                fig.savefig(os.path.join(fig_dir, layers_name + "_histogram.png"))

            layers_criticality[layers_names] = np.mean(criticality_values)
        ''' --------------------------------------- '''
        ''' Plotting models layers criticality '''
        ''' --------------------------------------- '''
        plot_models_models_criticality(layers_criticality,
                                       _path,
                                       _model_name,
                                       label,
                                       _cri_tau)


def plot_models_models_criticality(layer_criticality, _path, _model_name, _class, _tau):
    # ploting models layers criticality
    fig_dir = os.path.join(_path, _model_name, _class)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    colors = list()
    criticality = list()
    for each_layer, mean_criticality in layer_criticality.items():
        if mean_criticality > _tau:
            colors.append('red')
        else:
            colors.append('green')
        criticality.append(mean_criticality)

    x_pos = list(layer_criticality.keys())

    plt.rcParams.update({'font.size': 4})
    fig, ax = plt.subplots(figsize=(6, 3))

    ax.bar(x_pos, criticality, alpha=0.5, color=colors)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=45, horizontalalignment='right')
    plt.rcParams.update({'font.size': 6})
    ax.set_title("Layers\' normalized criticality for model: {}, for class: {}".format(_model_name, _class))
    ax.set_ylabel('Mean of normalized criticality')
    ax.set_xlabel('Layers\' names')
    # Tweak spacing to prevent clipping of tick-labels
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, _model_name + "_" + _class + "_criticality_result.pdf"))
