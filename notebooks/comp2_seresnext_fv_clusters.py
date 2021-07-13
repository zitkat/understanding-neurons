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
# # Comparative 2 Distribution of features in SEResNext
# Plots with common projection

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
# import umap
# reducer = umap.UMAP()

# %%
from utils import load_npy_fvs, split_seresnext_labels
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
# ## Load Data

# %%
stage = "initialized"

data, labels, _ = load_npy_fvs(data_path / f"{model_name}_{stage}" / "npys", mode=mode)
idata = data.reshape((data.shape[0], np.prod(data.shape[1:])))
iembedding = np.load((data_path / f"{model_name}_{stage}" / "emb_v1").with_suffix(".npy"))

# %%
stage = "pretrained"

data, labels, _ = load_npy_fvs(data_path / f"{model_name}_{stage}" / "npys", mode=mode)
pdata = data.reshape((data.shape[0], np.prod(data.shape[1:])))
pembedding = np.load((data_path / f"{model_name}_{stage}" / "emb_v1").with_suffix(".npy"))

# %%
stage = "finetuned"

data, labels, _ =  load_npy_fvs(data_path / f"{model_name}_{stage}" / "npys", mode=mode)
fdata = data.reshape((data.shape[0], np.prod(data.shape[1:])))
fembedding = np.load((data_path / f"{model_name}_{stage}" / "emb_v1").with_suffix(".npy"))
flipped_femb = np.array([1 , -1]) * fembedding

# %%
min_length = min(idata.shape[0], pdata.shape[0], fdata.shape[0])

# %%
all_data = np.concatenate([pdata[:min_length], fdata[:min_length], idata[:min_length]], axis=0)
all_labels = list(labels[:min_length]) * 3
all_names = min_length * ["initialized"] + min_length * ["pretrained"] + min_length * ["finetuned"] 

# %%
all_embedding = np.load((data_path / f"ipf_{model_name}_emb_v1").with_suffix(".npy"))

# %%
all_emb_df = split_seresnext_labels(pd.DataFrame(dict(x=all_embedding[:, 0], y=all_embedding[:, 1], network=all_names, label=all_labels)))
all_emb_df

# %% [markdown]
# ## Plots with common projection

# %%
fig, axs = plot_parametrized_var(all_emb_df, x_var="x", y_var="y",
                      row_var=None,
                      column_var="network", 
                      color_var="label", color_lab="Bottleneck",
                      mk_var=None,
                      display_colormarked_var=scatter_colormarked_var,
                      use_color_bar=True,
                      lines_leg_rect=[0.2, .095, 0.01, 0.01],
                      marks_leg_rect=[0.4, .095, 0.01, 0.01],
                      cbar_rect=[0.95, 0.15, 0.01, 0.7])

# %%
fig = plt.figure(stage, figsize=(15, 7))
s = sns.scatterplot(data=all_emb_df.iloc[::64],
                x = 'x', 
                y = 'y', 
                hue='label', 
                style='network',
                palette="viridis_r",
                )
h,l = s.get_legend_handles_labels()
print(len(h))
plt.legend(h[n_conv_layers + 1:],l[n_conv_layers + 1:])


norm = plt.Normalize(0, n_conv_layers)
sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
sm.set_array([])

plt.gca().figure.colorbar(sm).set_label("Layer depth")
