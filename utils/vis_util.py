#!python
# -*- coding: utf-8 -*-

__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import itertools


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


def plot_cdp_results(_path,
                     _cri_stat_dict,
                     _model_name,
                     _cri_tau=0.0):

    layer_criticality = list()
    list_of_layers = list()
    weak_hypothesis = list()
    top_x_neurons = "all"

    fig, ax = plt.subplots()

    for each_layer, filters_data in tqdm(_cri_stat_dict.items()):
        for labels_data in filters_data:
            kernel_indices = list(labels_data.keys())
            criticality_values = list()

            for kernel_index, criticality in labels_data.items():
                for label, label_criticality in criticality.items():

                    fig_dir = os.path.join(_path, _model_name, label, each_layer, "criticality")
                    if not os.path.exists(fig_dir):
                        os.makedirs(fig_dir)

                    if top_x_neurons is "all":
                        top_x_values = [float(value) for value in label_criticality]
                        plt.rcParams.update({'font.size': 10})
                    else:
                        indices = np.argsort(label_criticality)
                        top_x = indices[-top_x_neurons:]
                        top_x_values = [float(label_criticality[index]) for index in top_x]
                        plt.rcParams.update({'font.size': 12})

                    # criticality pro layer
                    only_critical_neurons = [float(criticality_value)
                                             for criticality_value in top_x_values if
                                             float(criticality_value) > _cri_tau]

                    list_of_layers.append(each_layer)
                    layer_criticality.append(np.mean(only_critical_neurons))
                    # layer_criticality.append( np.mean(only_critical_neurons) * (max_number_of_weights / len(top_x_values)) )

                    x_pos = np.arange(len(top_x_values))
                    colors = list()

                    for values in top_x_values:
                        if values > _cri_tau:
                            colors.append('red')
                            weak_hypothesis.append(each_layer)
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
                    ax.barh(x_pos, top_x_values, alpha=0.5, color=colors, align='center')
                    ax.invert_yaxis()  # labels read top-to-bottom
                    ax.set_xlabel('Criticality')
                    ax.set_ylabel('Indices')
                    #ax.set_title("Neural criticality for layer: {} and class: {}".format(each_layer, _class))

                    # fig.suptitle("Neurons\' criticality for class : pedestrian")
                    fig.tight_layout()
                    fig.savefig(os.path.join(fig_dir, each_layer + "_CNA_result.png"))

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
                    fig.savefig(os.path.join(fig_dir, each_layer + "_histogram.png"))
    ''' --------------------------------------- '''
    ''' Plotting models layers criticality '''
    ''' --------------------------------------- '''
    plot_models_models_criticality(list_of_layers,
                                   layer_criticality,
                                   weak_hypothesis,
                                   _path,
                                   _model_name)


def plot_models_models_criticality(_list_of_layers, layer_criticality, weak_hypothesis, _path, _model_name):
    # ploting models layers criticality
    fig_dir = os.path.join(_path, _model_name)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    colors = list()
    for each_layer in _list_of_layers:
        if each_layer in weak_hypothesis:
            colors.append('red')
        else:
            colors.append('green')

    x_pos = np.arange(len(_list_of_layers))

    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(figsize=(6,3))

    ax.bar(_list_of_layers, layer_criticality, alpha=0.5, color=colors)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=45, horizontalalignment='right')
    plt.rcParams.update({'font.size': 12})
    ax.set_title("Layers\' normalized criticality for model: {}, for class: {}".format(_model_name, _class))
    ax.set_ylabel('Mean of normalized criticality')
    ax.set_xlabel('Layers\' names')
    # plt.margins(0.2)
    # Tweak spacing to prevent clipping of tick-labels
    fig.tight_layout()
    # ax.subplots_adjust(bottom=0.25)
    fig.savefig(os.path.join(fig_dir, _model_name + "_criticality_result.pdf"))
