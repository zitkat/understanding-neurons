#!python
# -*- coding: utf-8 -*-

__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.offsetbox import (TextArea, OffsetImage,
                                  AnnotationBbox)
import seaborn as sns

from utils import plogger
from utils.vis_util import get_var_filter_iter, fill_dict, none2str

symbols = dict(zip(map(str, np.arange(0, 6, dtype=int)),
                   ["o", "d", "v", "^", "s", "p"]))


# %% Parametrized var plotting
def _plot_marked_var(df, mk_var,
                     x_var, y_var, mark_symbols=None,
                     fig=None, ax=None, **kwargs):
    if mark_symbols is None:
        mark_symbols = symbols
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1)

    if kwargs.pop("log_scale", False):
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=10)
    ax.grid(True)

    kwargs = kwargs.copy()
    alpha = kwargs.pop("alpha", 1.0)
    color = kwargs.pop("color", None)

    label_ext = str(kwargs.pop("label", ""))

    mark_vals = sorted(df[mk_var].unique())
    for mk in mark_vals:
        curr_df = df[df[mk_var] == mk]
        if color is not None:
            l, = ax.plot(curr_df[x_var], curr_df[y_var],
                         mark_symbols[mk], label=str(int(mk)) + label_ext,
                         color=color,
                         )
        else:
            l, = ax.plot(curr_df[x_var], curr_df[y_var],
                         mark_symbols[mk], label=str(int(mk)) + label_ext,
                         )

        ax.plot(curr_df[x_var], curr_df[y_var], label="",
                alpha=alpha, color=l.get_color(),
                )

    omarks = [Line2D([0], [0], marker=mark_symbols[o], color="grey")
              for o in mark_vals]
    return omarks


def plot_colormarked_var(ax, fig, df,
                         color_var, mk_var,
                         x_var, y_var,
                         **kwargs):
    """
    Use in plot_parametrized_var as display_colormarked_var function.
    """
    olines = []
    color_vals = df[color_var].unique()
    cm = kwargs.pop("color_map", plt.cm.viridis)  # TODO unify colormap treatment
    colors = cm(np.linspace(0, 1, len(color_vals)))[::-1]
    omarks = None
    for cc, color_val in enumerate(color_vals):
        omarks = _plot_marked_var(df[df[color_var] == color_val],
                                  mk_var=mk_var,
                                  x_var=x_var, y_var=y_var, fig=fig, ax=ax,
                                  color=colors[cc],
                                  label=" {}".format(color_val), **kwargs)
        olines += [Line2D([0], [0], color=colors[cc])]
    return color_vals, olines, omarks


def scatter_colormarked_var(ax, fig, df,
                            color_var, mk_var,
                            x_var, y_var,
                            **kwargs):
    """
    Use in plot_parametrized_var as display_colormarked_var function.
    """
    hue_style = fill_dict(["hue", "style"], [color_var, mk_var])
    s = sns.scatterplot(data=df,
                        **hue_style,
                        x=x_var,
                        y=y_var,
                        palette="viridis_r",  # TODO common color scaling across calls?
                        ax=ax)
    h, _ = s.get_legend_handles_labels()
    if mk_var is not None:
        mlen = len(df[mk_var].unique())
        omarks = h[-mlen:]
    else:
        mlen = 0
        omarks = []

    use_cbar = kwargs.get("use_color_bar", True)
    if not use_cbar:
        color_vals = df[color_var].unique()
        clen = len(color_vals)
        olines = h[:clen]
    else:
        color_vals = None
        norm = plt.Normalize(0, len(h) - mlen)
        olines = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)

    ax.legend().remove()
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])

    return color_vals, olines, omarks


def wrap_axes(axs, ncol, nrow):
    """
    Wraps axes in lists to allow nice iteration over rows and columns
    even where there is only one of them.
    """
    if nrow == 1 and ncol == 1:
        axs = [[axs]]
    elif nrow == 1:
        axs = [axs]
    elif ncol == 1:
        axs = [[ax] for ax in axs]
    return axs


def plot_parametrized_var(df : pd.DataFrame,
                          x_var, y_var,
                          column_var=None, row_var=None,
                          color_var=None, mk_var=None,
                          display_colormarked_var=plot_colormarked_var,
                          **kwargs):
    """
    This functions serves to display results of parametric study organized into pandas DataFrame,
    each column of the DataFrame can represent different dimension of visuailzatio, changing over:
    columns/rows of axes grid, colors/markers of plotted elements, and xy coordinates. With exception
    of x_var and y_var, variables are optional and can be omitted or replaced with None.

    :param df:
    :param x_var:
    :param y_var:
    :param column_var:
    :param row_var:
    :param color_var:
    :param mk_var:
    :param display_colormarked_var: a function
        (ax, fig, df, color_var, mk_var, x_var, y_var) -> color_vals, olines, omarks
        see scatter_colormarked_var or plot_colormarked_var
    :param kwargs: following kwargs are supported: x_lab, y_lab, column_lab, row_lab, color_lab,
        mk_lab, figsize;
        marks_leg_rect : for adjusting markers legend position , default [0.55, .07, 0.01, 0.01];
        lines_leg_rect : for adjusting lines legend position, default [0, .07, 0.01, 0.01];
        lines_ncol: number of columns in line legend;
        cbar_rect : for adjusting colorbar position, mutually exclusive with lines_leg_rect,
        default [0.805, 0.15, 0.01, 0.7];
        whatewer kwargs display_colormarked_var supports

    :return: figure and dictionary of axes indexed by (column, row) tuple
    """

    ncol, iter_columns = get_var_filter_iter(df, column_var)
    nrow, iter_rows = get_var_filter_iter(df, row_var)


    y_lab = kwargs.pop("y_lab", y_var)
    x_lab = kwargs.pop("x_lab", x_var)
    clm_lab = kwargs.pop("column_lab", none2str(column_var))
    row_lab = kwargs.pop("row_lab", none2str(row_var))
    cor_lab = kwargs.pop("color_lab", none2str(color_var))
    mk_lab = kwargs.pop("mk_lab", none2str(mk_var))

    marks_ax_rect = kwargs.pop("marks_leg_rect", [0.55, .07, 0.01, 0.01])
    lines_ax_rect = kwargs.pop("lines_leg_rect", [0, .07, 0.01, 0.01])
    lines_n_col = kwargs.pop("lines_ncol", 3)
    cbar_ax_rect = kwargs.pop("cbar_rect", [0.805, 0.15, 0.01, 0.7])

    figsize = kwargs.pop("figsize", (ncol * 4, nrow * 4))
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol,
                            figsize=figsize)
    fig.subplots_adjust(hspace=.2, wspace=.2)

    axs = wrap_axes(axs, ncol, nrow)
    axs_dict = {}

    # Call mark-color variable display for rows and columns
    for ii, (row_val, row_filt) in enumerate(iter_rows()):
        for jj, (col_val, clm_filt) in enumerate(iter_columns()):
            ax = axs[ii][jj]
            axs_dict[(row_val, col_val)] = ax


            # TODO add option to force axes extent

            ax.set_title(f"{row_lab}: {row_val}, {clm_lab}: {col_val}")
            color_vals, olines, omarks = \
                display_colormarked_var(ax, fig,
                                        df[clm_filt & row_filt],
                                        color_var, mk_var,
                                        x_var, y_var, **kwargs)
            if jj == 0:
                ax.set_ylabel(y_lab)
            if ii == nrow - 1:
                ax.set_xlabel(x_lab)

    if mk_var is not None:
        marks_ax = fig.add_axes(marks_ax_rect)
        marks_ax.set_axis_off()

        marks_ax.legend(handles=omarks,
                        labels=sorted(df[mk_var].unique()),
                        title=mk_lab, ncol=len(omarks),
                        borderaxespad=0., loc="upper center")
        axs_dict.update(dict(markers_legend=marks_ax))

    if color_var is not None:
        if color_vals is not None:
            lines_ax = fig.add_axes(lines_ax_rect)
            lines_ax.set_axis_off()
            lines_ax.legend(handles=olines,
                            labels=["{}".format(cval) for cval in color_vals],
                            title=cor_lab, ncol=lines_n_col,
                            borderaxespad=0., loc="upper center")
            axs_dict.update(dict(lines_legend=lines_ax))
        else:
            cbar_ax = fig.add_axes(cbar_ax_rect)
            fig.colorbar(olines, cax=cbar_ax).set_label(cor_lab)
            axs_dict.update(dict(cbar=cbar_ax))
    return fig, axs_dict

# %% Feature visualizations plots
def show_fvs(fv_array, ns, max_cols=4):
    """
    Show feature visualizations in grid.

    :param fv_array: array of feature visualizations shape (n, W, H, C), C =1|3
    :param ns: feature indices, e.g. channel or neuron indices
    :param max_cols: maximum columns to use
    :return:
    """

    N = len(ns)
    rows = max(int(np.ceil(N / max_cols)), 1)
    cols = min(N, max_cols)
    plogger.debug(rows, cols)
    fig, axs = plt.subplots(nrows=rows, ncols=cols,
                            figsize=(5*max_cols, 5*rows),
                            tight_layout=True)
    fig.subplots_adjust(wspace=0, hspace=1)
    for i, (n, ax) in enumerate(zip(ns, axs.flatten())):
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(n)
        ax.imshow(fv_array[i, ...], interpolation=None)
    return fig, axs


def get_fv_annotator(fig, ax):
    def annotator(imarr, xy, label,
                  im_offsets=(1, 1.25),
                  text_offsets=(0, .88)):
        im = OffsetImage(imarr, zoom=1)
        x = xy[0]
        y = xy[1]

        xbox = x + im_offsets[0]
        ybox = y + im_offsets[1]

        xtx = xbox + text_offsets[0]
        ytx = ybox + text_offsets[1]
        im.image.axes = ax

        iab = AnnotationBbox(im,
                             (x, y),
                             xybox=(xbox, ybox),
                             xycoords='data',
                             pad=0.3,
                             arrowprops=dict(arrowstyle="->"),
                             bboxprops={'edgecolor': "white"})
        tab = AnnotationBbox(TextArea(label),
                             (x, y),
                             xybox=(xtx, ytx),
                             xycoords='data',
                             bboxprops={'edgecolor': "white"}
                             )
        ax.add_artist(iab)
        ax.add_artist(tab)
    return annotator