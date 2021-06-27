#!python
# -*- coding: utf-8 -*-

__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"

from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

from lucent.optvis import render, param
from lucent.modelzoo.util import get_model_layers

from settings import transforms
from util import ncobj, batch_indices, now, ensured_path, get_layer, renderable_units
from visualizations import show_fvs
from mapped_model import build_layers_dict


def render_layer(model, layer, idcs, mode="neuron",
                 batch_size=6,
                 image_size=(50,),
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


def render_model(model, layers, idcs=None, mode="neuron",
                 outputs_path=None, output_suffix="",
                 batch_size=6,
                 image_size=(50,),
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
    all_layers = build_layers_dict(model)

    # REFACTOR move to function
    if layers == "all":
        print("Warning: rendering ALL layers, this might be caused by default "
              "value and will take really long!")
        selected_layers = all_layers
    elif callable(layers):
        selected_layers = OrderedDict((ln, lo) for ln, lo in all_layers.items() if layers(ln))
    elif isinstance(layers, list):
        selected_layers = OrderedDict((ln, all_layers[ln]) for ln in layers if ln in all_layers)
    elif isinstance(layers, str):
        selected_layers = OrderedDict((ln, lo) for ln, lo in all_layers.items() if layers in ln)
    else:
        raise ValueError("Unsupported specification of layers to render.")

    open(ensured_path(outputs_path / "layers.list"), "w", encoding="utf-8").write("\n".join(selected_layers.keys()))

    for layer_name, layer_object in selected_layers.items():
        n = renderable_units(layer_object)
        if n > 0:
            print(f"\n\n{now()} Starting layer {layer_name} - {mode}s\n")
        else:
            print(f"{now()} Skipping composite layer {layer_name} - {mode}s")
            continue

        ns = idcs
        if idcs is None:
            ns = list(range(n))

        # TODO save to h5
        output_npys_path = ensured_path((outputs_path / "npys") / (mode + "s-" + layer_name + "-" + output_suffix))
        output_fig_path = ensured_path((outputs_path / "figs") / (mode + "s-" + layer_name + "-" + output_suffix + ".png"))
        if output_npys_path.with_suffix(".npy").exists():
            print(f"{output_npys_path} already exists.")
            continue
        res = render_layer(model, layer_name, ns,
                           mode=mode,
                           batch_size=batch_size,
                           image_size=image_size,
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
        np.save(output_npys_path, res)
        f, a = show_fvs(res, ns, max_cols=8)
        f.savefig(output_fig_path)
        plt.close()
