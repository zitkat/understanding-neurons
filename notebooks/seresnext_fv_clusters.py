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
# # Distribution of fetures in SEResNext

# %%
import sys
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


# %%
def load_fvs(input_path: Path, mode="neurons", version="v1"):
    data_paths = list(input_path.glob(f"{mode}*{version}*.npy"))
    data_list = [np.load(dpath) for dpath in data_paths]
    data_labels = sum(([dpth.name[:-4],] * len(d) 
                       for dpth, d in zip(data_paths, data_list)),
                      start=[])
    data = np.concatenate(data_list, axis=0)
    data_labels = np.array(data_labels)
    return data, data_labels


# %%
model_name = "seresnext50_32x4d"

# %%
reducer = umap.UMAP()

# %% [markdown]
# ## Randomly initialized

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

# %%
n_conv_layers

# %%
stage = "initialized"
mode = "neurons"
data, labels = load_fvs(Path("data", stage + "_" + model_name, "npys"), mode=mode)
idata = data.reshape((data.shape[0], np.prod(data.shape[1:])))

# %%
idata.shape

# %%
# iembedding = reducer.fit_transform(idata)

# %%
iembedding = np.load(Path("data", 'initialized' + "_" + model_name, "emb_v1").with_suffix(".npy"))

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
mode="neurons"
data, labels = load_fvs(Path("data", stage + "_" + model_name, "npys"), mode=mode)
pdata = data.reshape((data.shape[0], np.prod(data.shape[1:])))

# %%
pdata.shape

# %%
labels.shape

# %%
# pembedding = reducer.fit_transform(pdata)

# %%
pembedding = np.load(Path("data", 'pretrained' + "_" + model_name, "emb_v1").with_suffix(".npy"))

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
data, labels = load_fvs(Path("data", stage + "_" + model_name, "npys"), mode=mode)
fdata = data.reshape((data.shape[0], np.prod(data.shape[1:])))

# %%
fdata.shape

# %%
labels.shape

# %%
# fembedding = reducer.fit_transform(fdata)

# %%
fembedding = np.load(Path("data", 'finetuned' + "_" + model_name, "emb_v1").with_suffix(".npy"))

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

# %% [markdown]
# ## Comparative plots

# %%
min_length = min(idata.shape[0], pdata.shape[0], fdata.shape[0])


# %%
def split_labels(df):
    df["label"] = df["label"].str.split("_").apply(lambda s: "_".join(s[1:]))
    df["bottleneck"] = df["label"].str.split("_").apply(lambda s: s[-3][-1] if len(s) >= 3 else "-")
    df["layer"] = df["label"].str.split("_").apply(lambda s: s[0])
    df["conv"] =  df["label"].str.split("_").apply(lambda s: s[-2][-1])
    return df


# %%
f_df = split_labels(pd.DataFrame(dict(x=flipped_femb[:min_length, 0], 
                                      y=flipped_femb[:min_length, 1], 
                                      label=labels[:min_length], 
                                      network="fine")))
p_df = split_labels(pd.DataFrame(dict(x=pembedding[:min_length, 0],
                                      y=pembedding[:min_length, 1], 
                                      label=labels[:min_length],
                                      network="pre")))
i_df = split_labels(pd.DataFrame(dict(x=iembedding[:min_length, 0], 
                                      y=iembedding[:min_length, 1], 
                                      label=labels[:min_length],
                                      network="init"
                                     )))

# %% [markdown]
# ### Pretrained vs finetuned

# %%
fig, (axl, axr) = plt.subplots(ncols=2, nrows=1, figsize=(22, 10))
axl.set_title("Pretrained")
sns.scatterplot(data=p_df,
                x = 'x', 
                y = 'y', 
                hue='label', 
                palette="viridis_r", 
                ax=axl)
axl.legend([])
axr.set_title("Finetuned")
sns.scatterplot(data=f_df,
                x = 'x', 
                y = 'y', 
                hue='label', 
                palette="viridis_r",
                ax=axr, )
plt.legend([])

# %% jupyter={"source_hidden": true}
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


# %%
comb_df = pd.concat((i_df, p_df, f_df))

# %%
from vizualizations import plot_parametrized_var, scatter_colormarked_var, plot_colormarked_var

# %%
plot_parametrized_var(comb_df, x_var="x", y_var="y",
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
fig, (axl, axm, axr) = plt.subplots(ncols=3, nrows=1, figsize=(22, 10))
axm.set_title("Pretrained")
sns.scatterplot(data=p_df,
                x = 'x', 
                y = 'y', 
                hue='bottleneck', 
                palette="viridis_r", 
                ax=axm)
# axm.legend([])

axl.set_title("Initialized")
sns.scatterplot(data=i_df,
                x = 'x', 
                y = 'y', 
                hue='bottleneck', 
                palette="viridis_r",
                ax=axl, )
# axl.legend([])
axr.set_title("Finetuned")
sns.scatterplot(data=f_df,
                x = 'x', 
                y = 'y', 
                hue='bottleneck', 
                palette="viridis_r",
                ax=axr, )
# plt.legend([])

# %% jupyter={"source_hidden": true}
fig, (axl, axm, axr) = plt.subplots(ncols=3, nrows=1, figsize=(22, 10))
axm.set_title("Pretrained")
sns.scatterplot(data=p_df,
                x = 'x', 
                y = 'y', 
                hue='conv', 
                palette="viridis_r", 
                ax=axm)
# axm.legend([])

axl.set_title("Initialized")
sns.scatterplot(data=i_df,
                x = 'x', 
                y = 'y', 
                hue='conv', 
                palette="viridis_r",
                ax=axl, )
# axl.legend([])
axr.set_title("Finetuned")
sns.scatterplot(data=f_df,
                x = 'x', 
                y = 'y', 
                hue='conv', 
                palette="viridis_r",
                ax=axr, )
# plt.legend([])

# %% [markdown]
# ## Plots with common projection

# %%


all_data = np.concatenate([pdata[:min_length], fdata[:min_length], idata[:min_length]], axis=0)
all_labels = list(labels[:min_length]) * 3
all_names = min_length * ["initialized"] + min_length * ["pretrained"] + min_length * ["finetuned"] 

# %%
# all_embedding = reducer.fit_transform(all_data)

# %%
all_embedding = np.load(Path("data", 'ipf' + "_" + model_name + "_" + "emb_v1").with_suffix(".npy"))

# %%
all_emb_df = split_labels(pd.DataFrame(dict(x=all_embedding[:, 0], y=all_embedding[:, 1], network=all_names, label=all_labels)))
all_emb_df

# %% [markdown]
# #### Save all embedding

# %%
np.save(Path("data", 'ipf' + "_" + model_name + "_" + "emb_v1"), all_embedding)

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

norm = plt.Normalize(0, n_conv_layers)
sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
sm.set_array([])
fig.subplots_adjust(wspace=0.025, right=0.8)
axc = fig.add_axes([0.805, 0.15, 0.01, 0.7])
axc.figure.colorbar(sm, cax=axc).set_label("Layer depth", fontsize=font)

# %%
fig = plt.figure(stage, figsize=(15, 7))
s = sns.scatterplot(data=all_emb_df.iloc[::64],
                x = 'x', 
                y = 'y', 
                hue='label', 
                style='network',
                palette="viridis_r",
                size="network",
                )
h,l = s.get_legend_handles_labels()
print(len(h))
plt.legend(h[n_conv_layers + 1:],l[n_conv_layers + 1:])


norm = plt.Normalize(0, n_conv_layers)
sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
sm.set_array([])

plt.gca().figure.colorbar(sm).set_label("Layer depth")

# %%
fig = plt.figure(stage, figsize=(30, 16))
sns.scatterplot(data=all_emb_df,
                x = 'x', 
                y = 'y', 
                hue='layer', 
                style='network',
                palette="viridis_r",
#                 size="network",
                )

# %%
fig = plt.figure(stage, figsize=(30, 16))
sns.scatterplot(data=all_emb_df,
                x = 'x', 
                y = 'y', 
                hue='bottleneck', 
                style='network',
                palette="viridis_r",
#                 size="network",
                )


# %%
fig = plt.figure(stage, figsize=(15, 10))
sns.scatterplot(data=all_emb_df,
                x = 'x', 
                y = 'y', 
                hue='conv', 
                style='network',
                palette="viridis_r",
#                 size="network",
                )


# %%
fig = plt.figure(stage, figsize=(30, 16))
sns.scatterplot(data=all_emb_df,
                x = 'x', 
                y = 'y', 
                hue='network',
                )
plt.gca().axis("off")

# %% [markdown]
# ## Show features by their projections

# %% [markdown]
# ### Initialized model

# %%
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox, BboxImage)


# %%
def get_ax_label(layer_name):
    parts = layer_name.split("_")
    if len(parts) <= 3:
        return parts[1]
    return f"L{parts[1][-1]}.{parts[2]} Conv {parts[3][-1]}"


# %%
def get_unit_from_labels(labels: np.ndarray, layer_name, n, prefix="neurons", suffix="v1"):
    wherelay = np.where(labels == "_".join((prefix, layer_name, suffix)))[0]
    lidx = wherelay[0]
    if  n > len(wherelay):
        raise ValueError(f"Only {len(wherelay)} in layer {layer_name} rendered but {n}-th one requested")
    nidx = lidx + n
    return nidx


# %%
def get_fv_annotator(fig, ax):
    
    def annotator(imarr, xy, label, 
                  im_offsets=(1, 1.25),
                  text_offsets=(0, .88)
                 ):
        im = OffsetImage(imarr, zoom=1)
        x = xy[0]
        y = xy[1]

        xbox = x + im_offsets[0]
        ybox = y + im_offsets[1]
        
        xtx = xbox + text_offsets[0]
        ytx = ybox + text_offsets[1]
        im.image.axes = ax

        iab = AnnotationBbox(im, 
                             [x, y],
                             xybox=(xbox, ybox),
                             xycoords='data',
                             pad=0.3,
                             arrowprops=dict(arrowstyle="->"),
                            bboxprops = {'edgecolor': "white"})
        tab = AnnotationBbox(TextArea(label),
                             [x, y],
                             xybox=(xtx, ytx),
                             xycoords='data',
                             bboxprops = {'edgecolor': "white"}
        )
        ax.add_artist(iab)
        ax.add_artist(tab)
    return annotator


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
         label=f"{get_ax_label(labels[uidx])}:{uidx % 64}",
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
             label=f"{get_ax_label(labels[uidx])}:{uidx % 64}",
             **udict)

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

# %% jupyter={"source_hidden": true}
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
             label=f"{get_ax_label(labels[uidx])}:{uidx % 64}",
             **udict)

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
             label=f"{get_ax_label(labels[uidx])}:{uidx % 64}",
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
             label=f"{get_ax_label(labels[uidx % len(labels)])}:{uidx % 64}",
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
             label=f"{get_ax_label(labels[uidx % len(labels)])}:{uidx % 64}",
             **udict)

norm = plt.Normalize(0, n_conv_layers)
sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
sm.set_array([])
fig.subplots_adjust(wspace=0.025, right=0.8)
axc = fig.add_axes([0.805, 0.15, 0.01, 0.7])
axc.figure.colorbar(sm, cax=axc).set_label("Layer depth", fontsize=font)

# %%
