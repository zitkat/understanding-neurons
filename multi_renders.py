#!python
# -*- coding: utf-8 -*-
"""
Functions for rendering: whole model or a single layer
"""

__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"

import time
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from lucent.optvis import render, param

from settings import transforms
from utils.model_util import ncobj, batch_indices, iterate_renderable_layers, build_layers_dict
from utils.process_util import plogger, ensured_path, now
from visualizations import show_fvs


def render_layer(model, layer, idcs, mode="neuron", batch_size=6, image_size=(50,),
                 # render_vis params
                 optimizer=None,
                 transforms=transforms,
                 thresholds=(512,),
                 verbose=False,
                 preprocess=True,
                 progress=True,
                 show_image=False,
                 save_image=False,
                 image_name=None,
                 show_inline=False,
                 fixed_image_size=None):
    res_list = []
    for indcs_batch in batch_indices(idcs, batch_size=batch_size):
        batch_param_f = lambda: param.image(*image_size, batch=len(indcs_batch))
        obj = sum([ncobj(mode, layer, n, b) for b, n in enumerate(indcs_batch)])
        res_list += render.render_vis(model, obj, batch_param_f,
                                      optimizer=optimizer,
                                      transforms=transforms,
                                      thresholds=thresholds,
                                      verbose=verbose,
                                      preprocess=preprocess,
                                      progress=progress,
                                      show_image=show_image,
                                      save_image=save_image,
                                      image_name=image_name,
                                      show_inline=show_inline,
                                      fixed_image_size=fixed_image_size,
                                      desc=f"{layer} | units: {indcs_batch} of {len(idcs)}"
                                      )
    return np.concatenate(res_list, axis=0)


def render_model(model, layers, idcs=None, mode="neuron", outputs_path=None,
                 output_suffix="", batch_size=6, image_size=(50,),
                 save_samples : bool = True, n_test_units : int = None,
                 stat_only : bool = False,
                 # render_vis params
                 optimizer=None,
                 transforms=transforms,
                 thresholds=(512,),
                 verbose=False,
                 preprocess=True,
                 progress=True,
                 show_image=False,
                 save_image=False,
                 image_name=None,
                 show_inline=False,
                 fixed_image_size=None):
    if hasattr(model, "module"):
        model = model.module
    model = model.to(0).eval()

    selected_layers = select_layers(layers, model)
    open(ensured_path(outputs_path / "layers.list"), "w", encoding="utf-8").write("\n".join(selected_layers.keys()))

    stats_list = []
    for layer_name, layer_object, n in iterate_renderable_layers(selected_layers,
                                                                 verbose=True):

        ns = idcs
        if idcs is None:
            ns = list(range(n))
            if stat_only:
                if n_test_units is not None:
                    ns = list(range(n_test_units))
                else:
                    ns = list(range(batch_size))

        # TODO save to h5
        output_npys_path = ensured_path((outputs_path / "npys") / (mode + "s-" + layer_name + "-" + output_suffix))
        output_fig_path = ensured_path((outputs_path / "figs") / (mode + "s-" + layer_name + "-" + output_suffix + ".png"))
        if output_npys_path.with_suffix(".npy").exists() and not stat_only:
            plogger.info(f"{output_npys_path} already exists.")
            continue

        t0 = time.time()
        res = render_layer(model, layer_name, ns, mode=mode, batch_size=batch_size, image_size=image_size,
                           optimizer=optimizer,
                           transforms=transforms,
                           thresholds=thresholds,
                           verbose=verbose,
                           preprocess=preprocess,
                           progress=progress,
                           show_image=show_image,
                           save_image=save_image,
                           image_name=image_name,
                           show_inline=show_inline,
                           fixed_image_size=fixed_image_size
                           )
        t1 = time.time()

        if save_samples:
            plogger.debug(f"{now()} Saving renders, shape: {res.shape}")
            np.save(output_npys_path, res)
            f, a = show_fvs(res, ns, max_cols=8)
            f.savefig(output_fig_path)
            plt.close()

        stats_dict = dict(name=layer_name, cls=type(layer_object),
                          units=n,
                          test_units=len(ns),
                          bs=batch_size,
                          unit_time=(t1 - t0) / len(ns),
                          batch_time=(t1 - t0) / len(ns) * batch_size,
                          all_time=(t1 - t0) / len(ns) * n
                          )
        stats_list.append(stats_dict)

    stats_df = pd.DataFrame(stats_list)
    stats_df.set_index("name", inplace=True)
    return stats_df


def select_layers(layers, model):
    all_layers = build_layers_dict(model)
    if layers == "all":
        plogger.warning("Warning: rendering ALL layers, this might be caused by default "
                        "value and will take really long!")
        selected_layers = all_layers
    elif callable(layers):
        selected_layers = OrderedDict(
                (ln, lo) for ln, lo in all_layers.items() if layers(ln))
    elif isinstance(layers, list):
        selected_layers = OrderedDict((ln, all_layers[ln]) for ln in layers if ln in all_layers)
    elif isinstance(layers, str):
        selected_layers = OrderedDict((ln, lo) for ln, lo in all_layers.items() if layers in ln)
    else:
        plogger.error(f"Unsupported specification of layers to render {layers}.")
        raise ValueError("Unsupported specification of layers to render.")
    return selected_layers
