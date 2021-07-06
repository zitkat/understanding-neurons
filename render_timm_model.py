#!python
# -*- coding: utf-8 -*-

__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"

from typing import List

import click
from pathlib import Path
import torch
import timm

from utils.model_util import get_timm_classfier
from utils.process_util import now, plogger, add_plog_file
from multi_renders import render_model
from settings import load_settings


@click.command(epilog='(c) 2021 T. Zitka, KKY UWB')
@click.argument("model-name")
@click.option("-w", "--model-weights", default="initialized",
              help="Can be 'pretrained', 'initialized' or path to pth file with state dict.")
@click.option("--layers", type=str, default="all",
              help="PAth .list file or regular expression selecting layers")
@click.option("--mode", "-m", multiple=True,
              help="Mode of rendering, neurons or channels.")
@click.option("-sv", "--settings-version", type=str, default="Default",
              help="Column in settings file to use as settings")
@click.option("--settings-file", type=Path, default=Path("settings.csv"))
@click.option("--output", type=Path, default=Path("data/renders"))
@click.option("--hide-progress", is_flag=True)
def main(model_name: str, model_weights: str,
         mode: List, layers: str,
         settings_version: str, settings_file: Path,
         output: Path,
         hide_progress: bool):

    add_plog_file(output / f"{now()}_{model_name}_{model_weights}.plog")
    plogger.info(f"Rendering {model_name}: {model_weights}")

    name = model_weights
    if model_weights.endswith(".pth"):
        model_weights = Path(model_weights)
        name = model_weights.stem
        model = get_timm_classfier(model_name, target_size=5)
        net_dict = torch.load(model_weights)
        model.load_state_dict(net_dict)
    elif model_weights == "pretrained":
        model = timm.create_model(model_name, pretrained=True)
    elif model_weights == "initialized":
        save_path = output / (model_name + "_init.pth")
        if save_path.exists():
            plogger.info(f"Loading existing initialization from {save_path}")
            model = get_timm_classfier(model_name, target_size=5)
            net_dict = torch.load(save_path)
            model.load_state_dict(net_dict)
        else:
            model = get_timm_classfier(model_name, target_size=5, pretrained=False)
            torch.save(model.cpu().state_dict(), save_path)
    else:
        plogger.error(f"Unknown option for model weights {model_weights} terminating!")
        return
    outputs_path = Path(output, model_name + "_" + name)

    if mode:
        modes = mode
    else:
        modes = ["neuron"]

    if layers.endswith(".list"):
        layers = open(Path(layers), "r").read().split("\n")

    settings = load_settings(settings_file, settings_version)

    for mode in modes:
        render_model(model,
                     layers=layers,
                     mode=mode,
                     outputs_path=outputs_path,
                     output_suffix=settings_version,
                     progress=not hide_progress,
                     **settings["render"])


if __name__ == '__main__':
    main()
