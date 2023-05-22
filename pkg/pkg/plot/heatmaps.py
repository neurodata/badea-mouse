from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.transforms import Bbox
from seaborn.utils import relative_luminance
from scipy.stats import rankdata

from ..data import GENOTYPES
from ..utils import squareize

np.seterr(all="ignore")
import warnings

# from pandas.core.common import SettingWithCopyWarning
# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

def plot_heatmaps(dfs, cbar=True, ranked_pvalue=False, top_ranks=10):
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
    if ranked_pvalue:
        # dfs.corrected_pvalue = rankdata(1 - dfs.corrected_pvalue, nan_policy='omit')
        
        tmp = []
        for idx, hier in enumerate(hiers):
            df = dfs[dfs.hierarchy_level == hier]
            df.loc[:,'corrected_pvalue'] = rankdata(1 - df.pvalue.values, method='max', nan_policy='omit')
            cmax = df.corrected_pvalue.max()
            df.loc[:, 'ranked_significant'] = df.loc[:,'corrected_pvalue'] > (cmax-top_ranks)
            tmp.append(df)
        dfs = pd.concat(tmp, axis=0,)
            
        # vmax = dfs.corrected_pvalue.max()
        vmax=None
        minimum = 1
    else:
        vmax = 1
        minimum = np.log10(dfs[(~dfs.corrected_pvalue.isna()) & (dfs.corrected_pvalue > 0)].corrected_pvalue).min()

    heatmap_kws = dict(
        cmap="RdBu",
        square=True,
        cbar=False,
        vmax=vmax,
        vmin=minimum,
        fmt="s",
        center=0,
    )
    
    for idx, hier in enumerate(hiers):
        df = dfs[dfs.hierarchy_level == hier]

        k = len(np.unique(df.region1))

        pvec = df.corrected_pvalue.values
        pvals = squareize(k, pvec)
        pvals[np.isnan(pvals)] = 1
        if ranked_pvalue:
            plot_pvalues = pvals
        else:
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
        
        if ranked_pvalue:
            zipped = zip(raveled_idx, (df.loc[:, ["significant", "ranked_significant"]].values))
        else:
            zipped=zip(raveled_idx, df.significant)
        for idx, is_significant in zipped:
            if ranked_pvalue:
                is_significant, is_ranked_significant = is_significant
            i, j = np.unravel_index(idx, (k, k))

            # REF: seaborn heatmap
            lum = relative_luminance(colors[idx])
            text_color = ".15" if lum > 0.408 else "w"
            lw = 20 / k
            
            if is_significant == True:
                if ranked_pvalue:
                    if is_ranked_significant == False:
                        continue
                    else:
                        xs = [j + pad, j + 1 - pad]
                        ys = [i + pad, i + 1 - pad]
                        ax.plot(xs, ys, color=text_color, linewidth=lw)
                        xs = [j + 1 - pad, j + pad]
                        ys = [i + pad, i + 1 - pad]
                        ax.plot(xs, ys, color=text_color, linewidth=lw)
                else:
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


def plot_pairwise(
    pairwise_df,
    ksample_df,
):
    n_plots = 3
    width_ratios = [0.5] + [4] * n_plots

    fig, axes = plt.subplots(
        ncols=4,
        nrows=1,
        figsize=(n_plots * 3.25 + 1, 3.8),
        dpi=300,
        gridspec_kw=dict(width_ratios=width_ratios),
        constrained_layout=True,
        # sharey=True,
    )

    # Subscript for ease
    cax = axes[0]
    axs = axes[1:]

    heatmap_kws = dict(
        cmap="RdBu",
        square=True,
        cbar=False,
        vmax=0,
        vmin=np.log10(pairwise_df[(~pairwise_df.corrected_pvalue.isna()) & (pairwise_df.corrected_pvalue > 0)].corrected_pvalue).min(),
        fmt="s",
        center=0,
    )

    genotype_pairs = list(combinations(GENOTYPES, 2))

    for idx, (genotype1, genotype2) in enumerate(genotype_pairs):
        df = pairwise_df[
            (pairwise_df.genotype1 == genotype1) & (pairwise_df.genotype2 == genotype2)
        ]

        k = len(np.unique(df.region1))

        pvec = df.corrected_pvalue.values
        pvals = squareize(k, pvec)
        pvals[np.isnan(pvals)] = 1
        pvals[pvals==0] = pvals[pvals > 0].min()
        plot_pvalues = np.log10(pvals)
        if np.isinf(plot_pvalues).any():
            plot_pvalues[np.isinf(plot_pvalues)] = -150

        # Labels for x y axis
        labels = list(pd.unique(df.region1))

        # make mask
        triu_idx = np.triu_indices_from(pvals)
        mask = np.ones((k, k))
        mask[triu_idx] = 0

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

        if idx > 0:
            ax.set_yticklabels("")
        ax.tick_params(
            axis="x",
            labelrotation=90,
            pad=0.5,
            length=1,
            bottom=False,
        )

        ax.set_title(f"{genotype1} vs. {genotype2}")

        # Make x's and o's
        colors = im.get_children()[0].get_facecolors()
        raveled_idx = np.ravel_multi_index(triu_idx, plot_pvalues.shape)
        pad = 0.2
        for idx, is_significant, ksample_significant in zip(
            raveled_idx,
            df.significant,
            ksample_df[ksample_df.hierarchy_level == 3].significant,
        ):
            i, j = np.unravel_index(idx, (k, k))

            # REF: seaborn heatmap
            lum = relative_luminance(colors[idx])
            text_color = ".15" if lum > 0.408 else "w"
            lw = 20 / k

            if ksample_significant == True:
                if is_significant == True:
                    xs = [j + pad, j + 1 - pad]
                    ys = [i + pad, i + 1 - pad]
                    ax.plot(xs, ys, color=text_color, linewidth=lw)
                    xs = [j + 1 - pad, j + pad]
                    ys = [i + pad, i + 1 - pad]
                    ax.plot(xs, ys, color=text_color, linewidth=lw)
                elif np.isnan(is_significant):
                    circ = plt.Circle(
                        (j + 0.5, i + 0.5),
                        0.25,
                        color=text_color,
                        linewidth=lw,
                        fill=False,
                    )
                    ax.add_artist(circ)
            # elif ksample_significant == False:
            #     triangle = RegularPolygon(
            #         (j + 0.5, i + 0.5),
            #         radius=.25,
            #         numVertices=3,
            #         color=text_color,
            #         linewidth=lw,
            #         fill=False,
            #     )
            #     ax.add_artist(triangle)

        ax.invert_xaxis()
        ax.invert_yaxis()

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
    # cax.annotate(
    #     r"$\alpha$",
    #     (0.05, np.log10(0.05)),
    #     xytext=(-20, -15),
    #     textcoords="offset points",
    #     va="center",
    #     ha="right",
    #     arrowprops={"arrowstyle": "-", "linewidth": 3, "relpos": (0, 0.5)},
    # )

    return fig, axes
