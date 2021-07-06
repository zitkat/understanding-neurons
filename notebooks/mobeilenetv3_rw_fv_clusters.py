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
# # Distribution of features in MobileNetv3 RW

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
model_name = "mobilenetv3_rw"
data_path = Path("../data/renders/")
mode = "neurons"

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
all_conv_layers = list(np.unique(labels))
n_conv_layers = len(all_conv_layers)

# %%
pembedding = reducer.fit_transform(pdata)
np.save(data_path / f"{model_name}_{stage}" / "pemb_v1", pembedding)

# %%
pembedding = np.load((data_path / f"{model_name}_{stage}" / "pemb_v1").with_suffix(".npy"))

# %%
fig = plt.figure(stage, figsize=(10, 10))
s = sns.scatterplot(x = pembedding[:, 0], y = pembedding[:, 1], hue=labels, palette="viridis_r")

h,l = s.get_legend_handles_labels()
print(len(h))
plt.legend(h[n_conv_layers:],l[n_conv_layers:],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


norm = plt.Normalize(0, n_conv_layers)
sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
sm.set_array([])

# Remove the legend and add a colorbar
# plt.gca().get_legend().remove()
plt.gca().figure.colorbar(sm).set_label("Layer depth")


# %%
def split_mobilenet_labels(df : pd.DataFrame, sep="-"):
    df["label"] = df["label"].str.split(sep).apply(lambda s: sep.join(s[1:]))
    df["se"] = df["label"].str.contains("se")
    df["block"] = df["label"].str.split(sep).apply(lambda s: s[1] if len(s) >= 3 else "-")
    df["branch"] = df["label"].str.split(sep).apply(lambda s: s[2] if len(s) >= 3 else "0")
    df["conv"] =  df["label"].str.split(sep).apply(lambda s: s[-2] if len(s) >= 3 else "-")
    return df


# %%
p_df = split_mobilenet_labels(pd.DataFrame(dict(x = pembedding[:, 0],
                                           y = pembedding[:, 1],
                                           label=labels)))

# %%
fig, axs = plot_parametrized_var(p_df[~p_df["se"]], x_var="x", y_var="y",
                                 row_var="block",
                                 column_var="branch", 
                                 color_var="label", color_lab="Depth",
                                 mk_var=None,
                                 display_colormarked_var=scatter_colormarked_var,
                                 use_color_bar=True,
                                 lines_leg_rect=[0.2, .095, 0.01, 0.01],
                                 marks_leg_rect=[0.4, .095, 0.01, 0.01],
                                 cbar_rect=[0.95, 0.15, 0.01, 0.7])
axs[("-", ("0"))].set_title("stem")

# %%
fig, axs = plot_parametrized_var(p_df[p_df["se"]], x_var="x", y_var="y",
                                 row_var="block",
                                 column_var="branch", 
                                 color_var="label", color_lab="Depth",
                                 mk_var=None,
                                 display_colormarked_var=scatter_colormarked_var,
                                 use_color_bar=True,
                                 lines_leg_rect=[0.2, .095, 0.01, 0.01],
                                 marks_leg_rect=[0.4, .095, 0.01, 0.01],
                                 cbar_rect=[0.95, 0.15, 0.01, 0.7])


# %%
