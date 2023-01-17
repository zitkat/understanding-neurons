# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Feature of critical units in MobileNetv3 RW

# %%
import sys

# %%
sys.path.append("..")
sys.path

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from pathlib import Path
import numpy as np
import pandas as pd

# %%
from utils import load_npy_fvs, split_mobilenet_labels, add_criticality_data
from visualizations import show_fvs

# %% [markdown]
# ### Constants

# %%
model_name = "mobilenetv3_rw"
data_path = Path("../data/renders/")
mode = "neurons"

# %% [markdown]
# ## Pretrained on ImageNet

# %%
stage = "pretrained"

# %%
data, labels, units = load_npy_fvs(data_path / f"{model_name}_{stage}" / "npys", mode=mode)
pdata = data.reshape((data.shape[0], np.prod(data.shape[1:])))

# %%
p_df = split_mobilenet_labels(pd.DataFrame(dict(imidx=range(data.shape[0]),
                                                unit=units,
                                                label=labels)))

# %% jupyter={"outputs_hidden": true}
p_df = add_criticality_data(p_df, from_json=Path("../data/criticals/mobilenetv3_rw/statistics_dict.json"))

# %%
data.shape

# %%
labels.shape

# %%
all_conv_layers = list(np.unique(labels))
n_conv_layers = len(all_conv_layers)

# %%
p_df.head()

# %%
tmp = p_df["mountain_bike"].sort_values()

# %%
tmp.plot()

# %%
p_df[p_df.mountain_bike > 0.5]

# %%
_ = show_fvs(data[p_df[p_df.mountain_bike > 0.5]["imidx"]],
         p_df[p_df.mountain_bike > 0.5].index,
         max_cols=8    
        )

# %%
data.shape

# %%
p_df[p_df.mountain_bike > 0.5].index

# %%
import numpy as np
X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
from sklearn.decomposition import NMF
model = NMF(n_components=2, init='random', random_state=0)
W = model.fit_transform(X)
H = model.components_

# %%
X.shape

# %%
W.shape

# %%
H.shape

# %%

import torch
import timm

from mapped_model import MappedModel


model = timm.create_model("mobilenetv3_rw", pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
mmodel = MappedModel(model).eval().to(device)

# %%
mmodel["blocks-5-2-conv_dw:332"].shape

# %%
from sklearn.decomposition import NMF
model = NMF(n_components=3, init='random', random_state=0)
W = model.fit_transform(X)
H = model.components_
