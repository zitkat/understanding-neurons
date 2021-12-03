#!python
# -*- coding: utf-8 -*-

__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"


import click
from pathlib import Path

from mapped_model import MappedModel
from utils import get_model

from utils.process_util import now, plogger, add_plog_file
from settings import load_settings

import torchvision


@click.command(epilog='(c) 2021 T. Zitka, KKY UWB')
@click.argument("model-name")
@click.argument("dataset", type=Path)
@click.option("-w", "--model-weights", default="initialized",
              help="Can be 'pretrained', 'initialized' or path to pth file with state dict.")
@click.option("--layers", type=str, default="all",
              help="Path .list file or regular expression selecting layers")
@click.option("-sv", "--settings-version", type=str, default="Default",
              help="Column in settings file to use as settings")
@click.option("--settings-file", type=Path, default=Path("rendering_settings.csv"))
@click.option("--output", type=Path, default=Path("data/records"))
@click.option("--hide-progress", is_flag=True)
def main(model_name: str,
         dataset: Path,
         model_weights: str,
         layers: str,
         settings_version: str, settings_file: Path,
         output: Path,
         hide_progress: bool = False
):
    """
    This script runs inference on the whole dataset and records activation
    statistics according to settings
    """
    add_plog_file(output / f"{now()}_{model_name}_{model_weights}_{dataset.name}.plog")
    plogger.info(f"Recording {model_name}: {model_weights} on {dataset.name}")

    model, name = get_model(model_name, model_weights, output)

    mmodel = MappedModel(model).to(0)

    dataset = torchvision.datasets.VOCSegmentation(root=r"D:\Datasets\PASCAL_VOC_2012",
                                                   year="2012")

    for sample in dataset:
        act = mmodel.forward(sample[0], return_activations=True)


        print(sample)



if __name__ == '__main__':
    main()
