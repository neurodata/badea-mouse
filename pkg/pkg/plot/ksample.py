import numpy as np
import matplotlib

import matplotlib.pyplot as plt 
import pandas as pd
from matplotlib.transforms import Bbox
import seaborn as sns
from seaborn.utils import relative_luminance

from ..utils import squareize

np.seterr(all="ignore")


def plot_heatmaps(dfs, cbar=True):
    hiers = np.unique(dfs.hierarchy_level)
    n_plots = round(len(hiers) / 2)

    width_ratios = [0.5] + [4] * n_plots

    fig, axes = plt.subplots(
        ncols=n_plots + 1,
        nrows=n_plots,
        figsize=(n_plots * 3.5 + 1, 3.8 * n_plots),
        dpi=300,
        gridspec_kw=dict(width_ratios=width_ratios),
        constrained_layout=True,
    )
    axs = axes[:, 1:].ravel()
    gs = axes[0, 0].get_gridspec()
    for ax in axes[:, 0]:
        ax.remove()

    cax = fig.add_subplot(gs[:, 0])

    # cax = axs[0]

    heatmap_kws = dict(
        cmap="RdBu",
        square=True,
        cbar=False,
        vmax=1,
        vmin=-150,
        fmt="s",
        center=0,
    )

    for idx, hier in enumerate(hiers):
        df = dfs[dfs.hierarchy_level == hier]

        k = len(np.unique(df.region1))

        pvec = df.corrected_pvalue.values
        pvals = squareize(k, pvec)
        pvals[np.isnan(pvals)] = 1
        plot_pvalues = np.log10(pvals)
        plot_pvalues[np.isinf(plot_pvalues)] = -150
        # plot_pvalues[np.isnan(plot_pvalues)] = 0

        # Labels for x y axis
        labels = list(pd.unique(df.region1))

        # make mask
        triu_idx = np.triu_indices_from(pvals)
        mask = np.ones((k, k))
        mask[triu_idx] = 0

        # Plot heatmap
        ax = axs[idx]
        im = sns.heatmap(
            plot_pvalues,
            mask=mask,
            ax=ax,
            yticklabels=labels,
            xticklabels=labels,
            **heatmap_kws,
        )
        ax.tick_params(
            axis="y",
            labelrotation=0,
            pad=0.5,
            length=1,
            left=False,
        )
        ax.tick_params(
            axis="x",
            labelrotation=90,
            pad=0.5,
            length=1,
            bottom=False,
        )
        # ax.tick_params(left=False, bottom=False, pad=0)

        # Make x's and o's
        colors = im.get_children()[0].get_facecolors()
        raveled_idx = np.ravel_multi_index(triu_idx, plot_pvalues.shape)
        pad = 0.2
        for idx, is_significant in zip(raveled_idx, df.significant):
            i, j = np.unravel_index(idx, (k, k))

            # REF: seaborn heatmap
            lum = relative_luminance(colors[idx])
            text_color = ".15" if lum > 0.408 else "w"
            lw = 20 / k
            if is_significant == True:
                xs = [j + pad, j + 1 - pad]
                ys = [i + pad, i + 1 - pad]
                ax.plot(xs, ys, color=text_color, linewidth=lw)
                xs = [j + 1 - pad, j + pad]
                ys = [i + pad, i + 1 - pad]
                ax.plot(xs, ys, color=text_color, linewidth=lw)
            elif np.isnan(is_significant):
                circ = plt.Circle(
                    (j + 0.5, i + 0.5), 0.25, color=text_color, linewidth=lw, fill=False
                )
                ax.add_artist(circ)

        ax.invert_xaxis()
        ax.invert_yaxis()

    if cbar:
        fig = axs[-1].get_figure()
        _ = fig.colorbar(
            im.get_children()[0],
            cax=cax,
            fraction=1,
            shrink=10,
            ticklocation="left",
        )
        cax.set_title(r"$log_{10}$" + "\ncorrected" "\np-value", pad=10)
        cax.plot(
            [0, 1],
            [np.log10(0.05), np.log10(0.05)],
            zorder=100,
            color="black",
            linewidth=3,
        )
        cax.annotate(
            r"$\alpha$",
            (0.05, np.log10(0.05)),
            xytext=(-20, -15),
            textcoords="offset points",
            va="center",
            ha="right",
            arrowprops={"arrowstyle": "-", "linewidth": 3, "relpos": (0, 0.5)},
        )
        # shrink_axis(cax, scale=0.8, shift=0.05)
    else:
        cax.remove()

    for i in range(len(hiers)):
        subplot_labels = ["(A)", "(B)", "(C)", "(D)"]
        axs[i].text(
            -0.05,
            1.05,
            subplot_labels[i],
            horizontalalignment="center",
            verticalalignment="center",
            transform=axs[i].transAxes,
            weight="bold",
        )

    return fig, axs