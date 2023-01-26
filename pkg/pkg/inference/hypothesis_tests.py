from itertools import combinations, combinations_with_replacement

import numpy as np
import pandas as pd
from hyppo.ksample import MANOVA, KSample
from scipy.stats import kruskal, mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests

def group_labels(
    vertex_labels,
):
    out = []

    uniques = np.unique(vertex_labels)

    for pair in combinations_with_replacement(uniques, 2):
        rdx = vertex_labels == pair[0]
        cdx = vertex_labels == pair[1]

        out.append(
            [pair, (rdx, cdx)],
        )

    return out



def run_ksample(
    graphs, vertex_labels, hierarchy_level, test="kruskal", absolute=False, reps=1000
):
    """
    hierarchy_level : int
        0 is top level, 1 is next, etc
    """
    num_graphs = len(graphs)

    pairs = group_labels(vertex_labels)

    out = []
    for pair, (rdx, cdx) in pairs:
        num_verts = np.max([rdx.sum(), cdx.sum()])
        triu_idx = np.triu_indices(num_verts, k=1)

        if not np.all(rdx == cdx):  # deal with off diagonal blocks
            to_test = [g[rdx, :][:, cdx].ravel() for g in graphs]
        else:  # when main diagonal block, only use upper triangle
            to_test = [g[rdx, :][:, cdx][triu_idx] for g in graphs]

        if absolute:
            to_test = [np.abs(t) for t in to_test]

        if to_test[0].size < 3:
            res = [np.nan, np.nan]
        else:
            try:
                if test.lower() == "dcorr":
                    res = KSample("Dcorr").test(*to_test, auto=False, reps=reps, workers=-1)
                elif test.lower() == "kruskal":
                    res = kruskal(*to_test)
                elif test.lower() == "manova":
                    res = MANOVA().test(*to_test)
            except:
                res = [np.nan, np.nan]
        # res = tester(*[np.abs(i) for i in to_test])
        stat, pval = res[:2]

        to_append = [*pair, stat, pval, hierarchy_level]
        out.append(to_append)

    columns = ["region1", "region2", "statistic", "pvalue", "hierarchy_level"]
    df = pd.DataFrame(out, columns=columns)

    non_nan = ~df.pvalue.isna()
    is_sig, corrected_pvalues, _, _ = multipletests(df[non_nan].pvalue, method="fdr_bh")
    df.loc[non_nan, "corrected_pvalue"] = corrected_pvalues
    df.loc[non_nan, "significant"] = is_sig

    return df


def run_pairwise(
    graphs, graph_labels, vertex_labels, test="wilcoxon", absolute=False, reps=1000
):
    num_graphs = len(graphs)
    pairs = group_labels(vertex_labels)

    graph_pairs = list(combinations(range(3), 2))

    out = []

    for graph1, graph2 in graph_pairs:
        gs = [graphs[graph1], graphs[graph2]]

        for pair, (rdx, cdx) in pairs:
            num_verts = np.max([rdx.sum(), cdx.sum()])
            triu_idx = np.triu_indices(num_verts, k=1)

            if not np.all(rdx == cdx):  # deal with off diagonal blocks
                to_test = [g[rdx, :][:, cdx].ravel() for g in gs]
            else:  # when main diagonal block, only use upper triangle
                to_test = [g[rdx, :][:, cdx][triu_idx] for g in gs]

            if absolute:
                to_test = [np.abs(t) for t in to_test]

            if to_test[0].size < 3:
                res = [np.nan, np.nan]
            else:
                try:
                    if test.lower() == "dcorr":
                        res = KSample("Dcorr").test(
                            *to_test, auto=False, reps=reps, workers=-1
                        )
                    elif test.lower() == "wilcoxon":
                        res = wilcoxon(*to_test)
                    elif test.lower() == "mannwhitney":
                        res = mannwhitneyu(*to_test)
                except:
                    res = [np.nan, np.nan]
            # res = tester(*[np.abs(i) for i in to_test])
            stat, pval = res[:2]

            to_append = [*pair, stat, pval, graph_labels[graph1], graph_labels[graph2]]
            out.append(to_append)

    columns = ["region1", "region2", "statistic", "pvalue", "genotype1", "genotype2"]
    df = pd.DataFrame(out, columns=columns)

    non_nan = ~df.pvalue.isna()
    is_sig, corrected_pvalues, _, _ = multipletests(df[non_nan].pvalue, method="fdr_bh")
    df.loc[non_nan, "corrected_pvalue"] = corrected_pvalues
    df.loc[non_nan, "significant"] = is_sig

    return df