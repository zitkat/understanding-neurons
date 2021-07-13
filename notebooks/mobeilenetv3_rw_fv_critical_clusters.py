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
from utils import load_npy_fvs, split_mobilenet_labels, add_criticality_data
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
data, labels, units = load_npy_fvs(data_path / f"{model_name}_{stage}" / "npys", mode=mode)
pdata = data.reshape((data.shape[0], np.prod(data.shape[1:])))

# %%
pdata.shape

# %%
labels.shape

# %%
all_conv_layers = list(np.unique(labels))
n_conv_layers = len(all_conv_layers)

# %%
# import umap
# reducer = umap.UMAP()
# pembedding = reducer.fit_transform(pdata)
# np.save(data_path / f"{model_name}_{stage}" / "pemb_v1", pembedding)

# %%
pembedding = np.load((data_path / f"{model_name}_{stage}" / "pemb_v1").with_suffix(".npy"))

# %%
fig = plt.figure(stage, figsize=(24, 20))
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
plt.show()

# %%
p_df = split_mobilenet_labels(pd.DataFrame(dict(x = pembedding[:, 0],
                                                y = pembedding[:, 1],
                                                unit=units,
                                                label=labels)))

# %% [markdown]
# ## Load criticalities
#
#
#

# %%
p_df = add_criticality_data(p_df, from_json=Path("../data/criticals/mobilenetv3_rw/statistics_dict.json"))

# %%
p_df = p_df.dropna()

# %%
fig, axs = plot_parametrized_var(p_df.dropna(), x_var="x", y_var="y",
                                 row_var="block",
                                 column_var="branch", 
                                 color_var="label", color_lab="Depth",
                                 mk_var="conv",
                                 display_colormarked_var=scatter_colormarked_var,
                                 use_color_bar=True,
                                 lines_leg_rect=[0.2, .095, 0.01, 0.01],
                                 marks_leg_rect=[0.4, .095, 0.01, 0.01],
                                 cbar_rect=[0.95, 0.15, 0.01, 0.7])
axs[("conv_stem", "0")].set_title("stem")
axs[("conv_head", "0")].set_title("head")

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
p_df["label"].unique()

# %%
