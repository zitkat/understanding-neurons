#!python
# -*- coding: utf-8 -*-

import json
import os

import torch


from mapped_model import MappedModel

from datasets.dataset_util import DataSet
from safety_util import SafetyAnalysis
from utils.vis_util import plot_cdp_results

if __name__ == '__main__':
    import timm

    models_to_evaluate = ["mobilenetv3_rw", "efficientnet_b0", "resnet18", "vit_base_patch16_224"]
    for models_name in models_to_evaluate:

        model = timm.create_model(models_name, pretrained=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        mmodel = MappedModel(model).eval().to(device)
        dataset = DataSet()

        dataset_path = os.path.join("data", "dataset", "imagenet", "train")
        output_path = os.path.join("data", "dataset", "imagenet", "classes")
        confidence_threshold = 0.8
        batch_size = 32
        dataset.sort_and_copy_images_according_to_label(mmodel,
                                               dataset_path,
                                               output_path,
                                               confidence_threshold,
                                               batch_size,
                                               device)

        dataset.load_testset_from_path(
            output_path,
            _resize=True,
            _normalize=True,
            _channels_last=False,
            _batch_size=batch_size,
            _width=224,
            _height=224,
            _channels=3
        )

        safety_analysis = SafetyAnalysis(mmodel, dataset)
        safety_analysis.analyse_criticality_via_plain_masking(device, models_name)
        with open(os.path.join("data", "statistics_dict.json")) as f:
            statistics = json.load(f)
            plot_cdp_results(os.path.join("data", "generated"), statistics, models_name, "all", 0.5)

