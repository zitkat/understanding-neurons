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
# import umap
# reducer = umap.UMAP()

# %%
from utils import load_npy_fvs, split_seresnext_labels, pretty_layer_label
from visualizations import plot_parametrized_var, scatter_colormarked_var, get_fv_annotator

# %% [markdown]
# ### Constants

# %%
model_name = "seresnext50_32x4d"
data_path = Path("../data/renders/")
mode = "neurons"

# %% jupyter={"source_hidden": true}
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

data, labels = load_npy_fvs(data_path / f"{model_name}_{stage}" / "npys", mode=mode)
idata = data.reshape((data.shape[0], np.prod(data.shape[1:])))
iembedding = np.load((data_path / f"{model_name}_{stage}" / "emb_v1").with_suffix(".npy"))

# %%
stage = "pretrained"

data, labels = load_npy_fvs(data_path / f"{model_name}_{stage}" / "npys", mode=mode)
pdata = data.reshape((data.shape[0], np.prod(data.shape[1:])))
pembedding = np.load((data_path / f"{model_name}_{stage}" / "emb_v1").with_suffix(".npy"))

# %%
stage = "finetuned"

data, labels =  load_npy_fvs(data_path / f"{model_name}_{stage}" / "npys", mode=mode)
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
# ## Show features by their projections

# %% [markdown]
# ### Initialized model

# %%
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox, BboxImage)


# %%
def closest(arr, coors):
    return np.argmin(np.sum((arr - coors)**2, axis=1))


# %%
uoi1 = np.argmax(iembedding[:, 1])

# %%
units_of_interest = [(np.argmax(iembedding[:, 1]), {}),
                     (np.argmin(iembedding[:, 1]), dict(im_offsets=(1.5, -1.5), text_offsets=(0, -.88))),
                     (np.argmin(iembedding[:, 0]), dict(im_offsets=(-1.5, 1))),
                     (np.argmax(iembedding[:, 0]), dict(im_offsets=(1.5, 4.25))),
                     (47, dict(im_offsets=(1.5, 4.5))),
                     (closest(iembedding, np.array([-3, -12])), dict(im_offsets=(-1, -2.65), text_offsets=(0, -.88))),
                     (closest(iembedding, np.array([-2.5, -12.5])), dict(im_offsets=(0, -2.15), text_offsets=(0, -.88)))
                    ]

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
s = sns.scatterplot(x = iembedding[:, 0], y = iembedding[:, 1], hue=labels, palette="viridis_r")

h,l = s.get_legend_handles_labels()
print(len(h))
plt.legend(h[n_conv_layers:], 
           l[n_conv_layers:],
           bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

norm = plt.Normalize(0, n_conv_layers)
sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
sm.set_array([])

# Remove the legend and add a colorbar
plt.gca().get_legend().remove()
# plt.gca().figure.colorbar(sm).set_label("Layer depth")

annotate = get_fv_annotator(fig, ax)
for uidx, udict in units_of_interest:
    annotate(idata[uidx].reshape(100, 100, 3), 
         iembedding[uidx],
         label=f"{pretty_layer_label(labels[uidx], sep='_')}:{uidx % 64}",
         **udict)

# %% [markdown]
# ### Pretrained

# %%
punits_of_interest = [(np.argmax(pembedding[:, 1]), dict(im_offsets=(0, 1.5), text_offsets=(0, 1.05))),
                      (np.argmin(pembedding[:, 1]), dict(im_offsets=(1.5, -1.5), text_offsets=(0, -.88))),
                      (np.argmin(pembedding[:, 0]), dict(im_offsets=(-3, 1))),
                      (np.argmax(pembedding[:, 0]), dict(im_offsets=(2., 1))),
                      (47, dict(im_offsets=(-3.2, -1.5))),
                      (closest(pembedding, np.array([-3, -12])), dict(im_offsets=(-1, -2.), text_offsets=(0, -.88))),
                      (closest(pembedding, np.array([2.75, 7])), dict(im_offsets=(-.5, -2.5), text_offsets=(0, -.88))),
                      (closest(pembedding, np.array([3, 7.2])), dict(im_offsets=(-3.5, -2.5), text_offsets=(0, -.88)))
                    ]

# %%
fig, ax = plt.subplots(1,1, figsize=(10, 10))
sns.scatterplot(x = pembedding[:, 0], y = pembedding[:, 1], hue=labels, palette="viridis_r")

h,l = s.get_legend_handles_labels()
print(len(h))
plt.legend(h[n_conv_layers:],l[n_conv_layers:],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


norm = plt.Normalize(0, n_conv_layers)
sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
sm.set_array([])

# Remove the legend and add a colorbar
# plt.gca().get_legend().remove()
# plt.gca().figure.colorbar(sm).set_label("Layer depth")

annotate = get_fv_annotator(fig, ax)
for uidx, udict in punits_of_interest:
    annotate(pdata[uidx].reshape(100, 100, 3), 
             pembedding[uidx],
             label=f"{pretty_layer_label(labels[uidx], sep='_')}:{uidx % 64}",
             **udict)

# %% [markdown]
# ## Finetuned

# %%
funits_of_interest = [(np.argmax(flipped_femb[:, 1]), dict(im_offsets=(0, 1.75), text_offsets=(0, 1.05))),
                      (np.argmin(flipped_femb[:, 1]), dict(im_offsets=(1.5, -2), text_offsets=(0, .88))),
                      (np.argmin(flipped_femb[:, 0]), dict(im_offsets=(-1.5, 1))),
                      (np.argmax(flipped_femb[:, 0]), dict(im_offsets=(1.2, 1))),
                      (47, dict(im_offsets=(-1.95, -1.1))),
                      (closest(flipped_femb, np.array([-3, -12])), dict(im_offsets=(0, -2.), text_offsets=(0, .88))),
                      (closest(flipped_femb, np.array([0, 3])), dict(im_offsets=(0, 3), text_offsets=(0, .88))),
                      (closest(flipped_femb, np.array([3, 7.2])), dict(im_offsets=(3.5, 2), text_offsets=(0, .88)))
                    ]

# %%
fig, ax = plt.subplots(1,1, figsize=(10, 10))
sns.scatterplot(x = flipped_femb[:, 0], 
                y = flipped_femb[:, 1], 
                hue=labels, palette="viridis_r")

h,l = s.get_legend_handles_labels()
print(len(h))
plt.legend(h[n_conv_layers:],l[n_conv_layers:],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


norm = plt.Normalize(0, n_conv_layers)
sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
sm.set_array([])

# Remove the legend and add a colorbar
# plt.gca().get_legend().remove()
# plt.gca().figure.colorbar(sm).set_label("Layer depth")

annotate = get_fv_annotator(fig, ax)
for uidx, udict in funits_of_interest:
    annotate(fdata[uidx].reshape(100, 100, 3), 
             flipped_femb[uidx],
             label=f"{pretty_layer_label(labels[uidx], sep='_')}:{uidx % 64}",
             **udict)

# %% [markdown]
# ## Together

# %%
fig, (axl, axm, axr) = plt.subplots(ncols=3, nrows=1, figsize=(22, 7))

font = 24

axl.set_title("Initialized", fontsize=font)
sns.scatterplot(data=all_emb_df[all_emb_df["network"] == "initialized"],
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

iunits_of_interest = [(all_emb_df[all_emb_df["network"] == "initialized"].x.argmax(), dict(im_offsets=(1, 6.), text_offsets=(0, 2.7))),
                      (all_emb_df[all_emb_df["network"] == "initialized"].y.argmax(), dict(im_offsets=(-10, 5.), text_offsets=(0, 2.7))),
                      (closest(all_emb_df[all_emb_df["network"] == "initialized"][["x", "y"]], np.array((5, 0))), dict(im_offsets=(-10, 2.), text_offsets=(0, 2.7))),
                      (closest(all_emb_df[all_emb_df["network"] == "initialized"][["x", "y"]], np.array((5, -2))), dict(im_offsets=(-9.9, -3), text_offsets=(0, 2.7))),
                      (closest(all_emb_df[all_emb_df["network"] == "initialized"][["x", "y"]], np.array((-25, -8))), dict(im_offsets=(3, 6.), text_offsets=(0, 2.7))),
                      (closest(all_emb_df[all_emb_df["network"] == "initialized"][["x", "y"]], np.array((-25, -9))), dict(im_offsets=(15, -2), text_offsets=(0, 2.7)))
                     ]
annotate = get_fv_annotator(fig, axl)
for uidx, udict in iunits_of_interest:
    annotate(idata[uidx % len(labels)].reshape(100, 100, 3), 
             all_emb_df.loc[uidx, ["x", "y"]],
             label=f"{pretty_layer_label(labels[uidx], sep='_')}:{uidx % 64}",
             **udict)


axm.set_title("Pretrained", fontsize=font)
sns.scatterplot(data=all_emb_df[all_emb_df["network"] == "pretrained"],
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
axm.set_xlim(axl.get_xlim())
axm.set_ylim(axl.get_ylim())

iunits_of_interest = [(all_emb_df[all_emb_df["network"] == "pretrained"].x.argmax(), dict(im_offsets=(1, 5.), text_offsets=(0, 2.7))),
                      (all_emb_df[all_emb_df["network"] == "pretrained"].y.argmax(), dict(im_offsets=(-11, 5.1), text_offsets=(0, 2.7))),
                      (closest(all_emb_df[all_emb_df["network"] == "pretrained"][["x", "y"]], np.array((5, 0))), dict(im_offsets=(-10, 2.), text_offsets=(0, 2.7))),
                      (closest(all_emb_df[all_emb_df["network"] == "pretrained"][["x", "y"]], np.array((5, -2))), dict(im_offsets=(-9.9, -3), text_offsets=(0, 2.7))),
                      (closest(all_emb_df[all_emb_df["network"] == "pretrained"][["x", "y"]], np.array((-25, -8))), dict(im_offsets=(3, 6.), text_offsets=(0, 2.7))),
                      (closest(all_emb_df[all_emb_df["network"] == "pretrained"][["x", "y"]], np.array((-25, -9))), dict(im_offsets=(15, -2), text_offsets=(0, 2.7)))
                     ]
annotate = get_fv_annotator(fig, axm)
for uidx, udict in iunits_of_interest:
    annotate(pdata[uidx % len(labels)].reshape(100, 100, 3), 
             all_emb_df[all_emb_df["network"] == "pretrained"].iloc[uidx][["x", "y"]],
             label=f"{pretty_layer_label(labels[uidx % len(labels)], sep='_')}:{uidx % 64}",
             **udict)


axr.set_title("Finetuned", fontsize=font)
s = sns.scatterplot(data=all_emb_df[all_emb_df["network"] == "finetuned"],
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
axr.set_xlim(axl.get_xlim())
axr.set_ylim(axl.get_ylim())

iunits_of_interest = [(all_emb_df[all_emb_df["network"] == "finetuned"].x.argmax(), dict(im_offsets=(12, 7.), text_offsets=(0, 2.7))),
                      (all_emb_df[all_emb_df["network"] == "finetuned"].y.argmax(), dict(im_offsets=(0, 4), text_offsets=(0, 2.7))),
                      (closest(all_emb_df[all_emb_df["network"] == "finetuned"][["x", "y"]], np.array((-25, -8))), dict(im_offsets=(15, 3.), text_offsets=(0, 2.7))),
                      (closest(all_emb_df[all_emb_df["network"] == "finetuned"][["x", "y"]], np.array((-25, -9))), dict(im_offsets=(15, -2), text_offsets=(0, 2.7)))
                     ]
annotate = get_fv_annotator(fig, axr)
for uidx, udict in iunits_of_interest:
    print(uidx)
    annotate(fdata[uidx % len(labels)].reshape(100, 100, 3), 
             all_emb_df[all_emb_df["network"] == "finetuned"].iloc[uidx][["x", "y"]],
             label=f"{pretty_layer_label(labels[uidx % len(labels)], sep='_')}:{uidx % 64}",
             **udict)

norm = plt.Normalize(0, n_conv_layers)
sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
sm.set_array([])
fig.subplots_adjust(wspace=0.025, right=0.8)
axc = fig.add_axes([0.805, 0.15, 0.01, 0.7])
axc.figure.colorbar(sm, cax=axc).set_label("Layer depth", fontsize=font)

# %%
