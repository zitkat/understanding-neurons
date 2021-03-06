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
# # Comparative 1 Distribution of features in SEResNext

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

# %% [markdown]
# ## Create DataFrames

# %%
min_length = min(idata.shape[0], pdata.shape[0], fdata.shape[0])

# %%
f_df = split_seresnext_labels(pd.DataFrame(dict(x=flipped_femb[:min_length, 0],
                                      y=flipped_femb[:min_length, 1], 
                                      label=labels[:min_length], 
                                      network="fine")))
p_df = split_seresnext_labels(pd.DataFrame(dict(x=pembedding[:min_length, 0],
                                      y=pembedding[:min_length, 1], 
                                      label=labels[:min_length],
                                      network="pre")))
i_df = split_seresnext_labels(pd.DataFrame(dict(x=iembedding[:min_length, 0], 
                                      y=iembedding[:min_length, 1], 
                                      label=labels[:min_length],
                                      network="init"
                                     )))

# %%
comb_df = pd.concat((i_df, p_df, f_df))

# %% [markdown]
# ## Simple plot using seaborn scatter plot directly

# %%
fig, (axl, axm, axr) = plt.subplots(ncols=3, nrows=1, figsize=(22, 7))

font = 24

axl.set_title("Initialized", fontsize=font)
sns.scatterplot(data=i_df,
                x = 'x', 
                y = 'y', 
                hue='label', 
                palette="viridis_r",
                ax=axl, )
axl.legend().remove()
axl.set_xlabel(None)
axl.set_ylabel(None)
axl.axes.xaxis.set_ticks([])
axl.axes.yaxis.set_ticks([])


axm.set_title("Pretrained", fontsize=font)
sns.scatterplot(data=p_df,
                x = 'x', 
                y = 'y', 
                hue='label', 
                palette="viridis_r", 
                ax=axm)
axm.legend().remove()
axm.set_xlabel(None)
axm.set_ylabel(None)
axm.axes.xaxis.set_ticks([])
axm.axes.yaxis.set_ticks([])

axr.set_title("Finetuned", fontsize=font)
s = sns.scatterplot(data=f_df,
                x = 'x', 
                y = 'y', 
                hue='label', 
                palette="viridis_r",
                ax=axr, )
axr.legend().remove()
axr.set_xlabel(None)
axr.set_ylabel(None)
axr.axes.xaxis.set_ticks([])
axr.axes.yaxis.set_ticks([])

norm = plt.Normalize(0, n_conv_layers)
sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
sm.set_array([])
fig.subplots_adjust(wspace=0.025, right=0.8)
axc = fig.add_axes([0.805, 0.15, 0.01, 0.7])
axc.figure.colorbar(sm, cax=axc).set_label("Layer depth", fontsize=font)


# %% [markdown]
# ## Using parametrized variable visualization

# %%
fig, axs = plot_parametrized_var(comb_df, x_var="x", y_var="y",
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
fig, axs = plot_parametrized_var(comb_df[comb_df["network"].isin(["pre", "fine"])], 
                      x_var="x", y_var="y",
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
fig, axs = plot_parametrized_var(comb_df, x_var="x", y_var="y",
                      row_var="layer",
                      column_var="network", 
                      color_var="bottleneck", color_lab="Bottleneck",
                      mk_var="conv",
                      display_colormarked_var=scatter_colormarked_var,
                      use_color_bar=True,
                      lines_leg_rect=[0.2, .095, 0.01, 0.01],
                      marks_leg_rect=[0.4, .095, 0.01, 0.01],
                      cbar_rect=[0.95, 0.15, 0.01, 0.7])

# %%
fig, axs = plot_parametrized_var(comb_df, x_var="x", y_var="y",
                      row_var=None,
                      column_var="network", 
                      color_var="bottleneck", color_lab="Bottleneck",
                      mk_var="conv",
                      display_colormarked_var=scatter_colormarked_var,
                      use_color_bar=True,
                      lines_leg_rect=[0.2, .095, 0.01, 0.01],
                      marks_leg_rect=[0.4, .095, 0.01, 0.01],
                      cbar_rect=[0.95, 0.15, 0.01, 0.7])

# %%
fig, axs = plot_parametrized_var(comb_df, x_var="x", y_var="y",
                      row_var=None,
                      column_var="network", 
                      color_var="conv", color_lab="Conv",
                      mk_var=None,
                      display_colormarked_var=scatter_colormarked_var,
                      use_color_bar=True,
                      lines_leg_rect=[0.2, .095, 0.01, 0.01],
                      marks_leg_rect=[0.4, .095, 0.01, 0.01],
                      cbar_rect=[0.95, 0.15, 0.01, 0.7])
