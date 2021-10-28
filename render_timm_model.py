#!python
# -*- coding: utf-8 -*-

__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"

from typing import List

import click
from pathlib import Path
import torch
import timm

from utils.model_util import get_timm_classfier, get_model
from utils.process_util import now, plogger, add_plog_file
from multi_renders import render_model
from settings import load_settings


@click.command(epilog='(c) 2021 T. Zitka, KKY UWB')
@click.argument("model-name")
@click.option("-w", "--model-weights", default="initialized",
              help="Can be 'pretrained', 'initialized' or path to pth file with state dict.")
@click.option("--layers", type=str, default="all",
              help="Path .list file or regular expression selecting layers")
@click.option("--mode", "-m", multiple=True,
              help="Mode of rendering, neurons or channels.")
@click.option("-sv", "--settings-version", type=str, default="Default",
              help="Column in settings file to use as settings")
@click.option("--settings-file", type=Path, default=Path("rendering_settings.csv"))
@click.option("--output", type=Path, default=Path("data/renders"))
@click.option("--hide-progress", is_flag=True)
@click.option("--stat-only", is_flag=True)
@click.option("--no-save-samples", is_flag=True)
def main(model_name: str, model_weights: str,
         mode: List, layers: str,
         settings_version: str, settings_file: Path,
         output: Path,
         hide_progress: bool, stat_only : bool, no_save_samples : bool):
    """
    This script renders feature visualizations of a timm model using lucent
    """

    add_plog_file(output / f"{now()}_{model_name}_{model_weights}.plog")
    plogger.info(f"Rendering {model_name}: {model_weights}")

    model, name = get_model(model_name, model_weights, output)
    outputs_path = Path(output, model_name + "_" + name)

    if mode:
        modes = mode
    else:
        modes = ["neuron"]

    if layers.endswith(".list"):
        layers = open(Path(layers), "r").read().split("\n")

    settings = load_settings(settings_file, settings_version)

    for mode in modes:
        df = render_model(model,
                          layers=layers,
                          mode=mode,
                          outputs_path=outputs_path,
                          output_suffix=settings_version,
                          progress=not hide_progress,
                          stat_only=stat_only,
                          save_samples=not no_save_samples,
                          **settings["render"])
        df.to_csv(outputs_path / f"{mode}s-stats.csv")


if __name__ == '__main__':
    main()
