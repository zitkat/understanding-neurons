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
# # Distribution of features in SEResNext

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
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
import seaborn as sns
import pandas as pd

# %%
import umap
reducer = umap.UMAP()

# %%
from utils import load_npy_fvs
from visualizations import plot_parametrized_var, scatter_colormarked_var

# %% [markdown]
# ### Constants

# %%
model_name = "seresnext50_32x4d"
data_path = Path("../data/renders/")
mode = "neurons"

# %%
all_conv_layers = ['conv1', 
                   
                   'layer1_0_conv1', 'layer1_0_conv2', 'layer1_0_conv3', 'layer1_1_conv1', 
                   'layer1_1_conv2', 'layer1_1_conv3', 'layer1_2_conv1', 'layer1_2_conv2', 'layer1_2_conv3', 
                   
                   'layer2_0_conv1', 'layer2_0_conv2', 'layer2_0_conv3', 'layer2_1_conv1', 'layer2_1_conv2', 
                   'layer2_1_conv3', 'layer2_2_conv1', 'layer2_2_conv2', 'layer2_2_conv3', 'layer2_3_conv1', 
                   'layer2_3_conv2', 'layer2_3_conv3', 
                   
                   'layer3_0_conv1', 'layer3_0_conv2', 'layer3_0_conv3', 
                   'layer3_1_conv1', 'layer3_1_conv2', 'layer3_1_conv3', 'layer3_2_conv1', 'layer3_2_conv2', 
                   'layer3_2_conv3', 'layer3_3_conv1', 'layer3_3_conv2', 'layer3_3_conv3', 'layer3_4_conv1', 
                   'layer3_4_conv2', 'layer3_4_conv3', 'layer3_5_conv1', 'layer3_5_conv2', 'layer3_5_conv3', 
                   
                   'layer4_0_conv1', 'layer4_0_conv2', 'layer4_0_conv3', 'layer4_1_conv1', 'layer4_1_conv2', 
                   'layer4_1_conv3', 'layer4_2_conv1', 'layer4_2_conv2', 'layer4_2_conv3']

n_conv_layers = len(all_conv_layers)
n_conv_layers

# %% [markdown]
# ## Randomly initialized

# %%
stage = "initialized"

# %%
data, labels = load_npy_fvs(data_path / f"{model_name}_{stage}" / "npys", mode=mode)
idata = data.reshape((data.shape[0], np.prod(data.shape[1:])))

# %%
idata.shape

# %%
# iembedding = reducer.fit_transform(idata)

# %%
iembedding = np.load((data_path / f"{model_name}_{stage}" / "emb_v1").with_suffix(".npy"))

# %%
fig = plt.figure(stage, figsize=(10, 10))
s = sns.scatterplot(x = iembedding[:, 0], 
                    y = iembedding[:, 1], 
                    hue=labels, 
                    palette="viridis_r")

h,l = s.get_legend_handles_labels()
print(len(h))
plt.legend(h[n_conv_layers:],
           l[n_conv_layers:],
           bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


norm = plt.Normalize(0, n_conv_layers)
sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
sm.set_array([])

# Remove the legend and add a colorbar
# plt.gca().get_legend().remove()
plt.gca().figure.colorbar(sm).set_label("Layer depth")

# %% [markdown]
# ## Pretrained on ImageNet

# %%
stage = "pretrained"

# %%
data, labels = load_npy_fvs(data_path / f"{model_name}_{stage}" / "npys", mode=mode)
pdata = data.reshape((data.shape[0], np.prod(data.shape[1:])))

# %%
pdata.shape

# %%
labels.shape

# %%
# pembedding = reducer.fit_transform(pdata)

# %%
pembedding = np.load((data_path / f"{model_name}_{stage}" / "emb_v1").with_suffix(".npy"))

# %%
fig = plt.figure(stage, figsize=(10, 10))
sns.scatterplot(x = pembedding[:, 0], y = pembedding[:, 1], hue=labels, palette="viridis_r")

h,l = s.get_legend_handles_labels()
print(len(h))
plt.legend(h[n_conv_layers:],l[n_conv_layers:],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


norm = plt.Normalize(0, n_conv_layers)
sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
sm.set_array([])

# Remove the legend and add a colorbar
# plt.gca().get_legend().remove()
plt.gca().figure.colorbar(sm).set_label("Layer depth")

# %% [markdown]
# ## Finetuned for specififc task

# %%
stage = "finetuned"

# %%
data, labels =  load_npy_fvs(data_path / f"{model_name}_{stage}" / "npys", mode=mode)
fdata = data.reshape((data.shape[0], np.prod(data.shape[1:])))

# %%
fdata.shape

# %%
labels.shape

# %%
# fembedding = reducer.fit_transform(fdata)

# %%
fembedding = np.load((data_path / f"{model_name}_{stage}" / "emb_v1").with_suffix(".npy"))

# %%
flipped_femb = np.array([1 , -1]) * fembedding

# %%
fig = plt.figure(stage, figsize=(10, 10))
sns.scatterplot(x = flipped_femb[:, 0], y = flipped_femb[:, 1], hue=labels, palette="viridis_r")

h,l = s.get_legend_handles_labels()
print(len(h))
plt.legend(h[n_conv_layers:],l[n_conv_layers:],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


norm = plt.Normalize(0, n_conv_layers)
sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
sm.set_array([])

# Remove the legend and add a colorbar
# plt.gca().get_legend().remove()
plt.gca().figure.colorbar(sm).set_label("Layer depth")
