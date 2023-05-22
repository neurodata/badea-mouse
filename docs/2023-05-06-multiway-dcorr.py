# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hyppo.independence import Dcorr
from sklearn.preprocessing import OneHotEncoder

from statsmodels.stats.multitest import multipletests
from itertools import combinations

# %%
volumes = pd.read_csv("../data/new/processed/volumes.csv")
meta = pd.read_csv("../data/new/processed/meta.csv")

meta.head()

vols = []
for idx, row in meta.iterrows():
    ID = row["ID"]
    vols.append(volumes[[ID]].values.ravel())

vols = np.vstack(vols)
vols /= vols.sum(axis=1, keepdims=True)

# %%

# Column labels
# APOE2, APOE3, APOE4, NON-HN, HN, Female, Male
enc = OneHotEncoder(handle_unknown="ignore")
tmp = meta[["Genotype", "Sex", "Diet", "Allele"]]
enc.fit(tmp)

multiway_table = enc.transform(tmp).toarray()

enc.get_feature_names_out()

lookup = {key: val for key, val in zip(enc.get_feature_names_out(), range(12))}

# %%
groups = [
    ["Sex_Female", "Sex_Male", "Diet_Control", "Allele_HN", "Allele_Non-HN"],
    ["Sex_Female", "Sex_Male", "Diet_Control"],
    ["Diet_Control", "Allele_HN", "Allele_Non-HN"],
    ["Diet_Control"],
]

genotype_groups = []
for i in [3, 2]:
    genotype_groups += list(
        combinations(["Genotype_APOE22", "Genotype_APOE33", "Genotype_APOE44"], i)
    )
genotype_groups = [list(i) for i in genotype_groups]

TEST_LIST = [i + j for i in genotype_groups for j in groups]

# %%

res = []

for test in TEST_LIST:
    lookup_codes = [lookup[i] for i in test]
    test_multiway_table = multiway_table[:, lookup_codes]
    maximum = np.max(test_multiway_table.sum(axis=1))

    idx = test_multiway_table.sum(axis=1) == maximum

    x = vols[idx]
    y = test_multiway_table[idx]

    stat, pval = Dcorr().test(x, y)

    apoe2 = True if "Genotype_APOE22" in test else False
    apoe3 = True if "Genotype_APOE33" in test else False
    apoe4 = True if "Genotype_APOE44" in test else False
    sex = True if "Sex_Female" in test else False
    allele = True if "Allele_HN" in test else False
    multiway = True if sex or allele else False

    res.append([multiway, apoe2, apoe3, apoe4, sex, allele, pval])


# %%
columns = ["Multiway", "APOE2", "APOE3", "APOE4", "Sex", "Allele", "pval"]
df = pd.DataFrame(res, columns=columns)

df["K"] = (
    (df[["APOE2", "APOE3", "APOE4"]].sum(axis=1)) * (df["Sex"] + 1) * (df["Allele"] + 1)
)

df.sort_values(by="K", inplace=True, ascending=False)

i_new = []
for idx, row in df.iterrows():
    index = ""
    for col in ["K", "Multiway", "APOE2", "APOE3", "APOE4", "Sex", "Allele"]:
        if col == "K":
            index += str(row[col]) + " | "
        else:
            if row[col]:
                index += "X | "
            else:
                index += "  | "
    i_new.append(index)

df.index = i_new

_, corrected_pvals, _, alpha = multipletests(
    df[["pval"]].values.ravel(), method="bonferroni"
)
df["corrected_pval"] = corrected_pvals

pvalues = df.copy()[["corrected_pval"]]
mask = np.select([pvalues < alpha, pvalues >= alpha], ["X", ""], default=pvalues)

df.to_csv("../outs/2021-05-06-multiway-dcorr-whole-vols.csv")

# %%
fig, ax = plt.subplots(figsize=(2, 4), dpi=300)

sns.heatmap(
    pvalues.transform("log10"),
    ax=ax,
    annot=mask,
    fmt="",
    square=False,
    linewidths=0.5,
    center=alpha,
    cbar_kws={"ticks": np.log10([0.01, 0.05, 0.1, 1])},
    cmap="RdBu",
)

ax.collections[0].colorbar.set_label("pvalue (log scale, bonferroni-adjusted)")
ax.collections[0].colorbar.set_ticklabels([0.01, 0.05, 0.1, 1])

ax.set_yticklabels(ax.get_yticklabels(), ha="right", fontdict={"family": "monospace"})

# %%
# Run node wise

res = []

for test in TEST_LIST:
    lookup_codes = [lookup[i] for i in test]
    test_multiway_table = multiway_table[:, lookup_codes]
    maximum = np.max(test_multiway_table.sum(axis=1))

    idx = test_multiway_table.sum(axis=1) == maximum

    x = vols[idx]
    y = test_multiway_table[idx]

    tmp_res = []
    for i in range(x.shape[1]):
        stat, pval = Dcorr().test(x[:, i], y)
        tmp_res.append(pval)

    apoe2 = True if "Genotype_APOE22" in test else False
    apoe3 = True if "Genotype_APOE33" in test else False
    apoe4 = True if "Genotype_APOE44" in test else False
    sex = True if "Sex_Female" in test else False
    allele = True if "Allele_HN" in test else False
    multiway = True if sex or allele else False

    res.append([multiway, apoe2, apoe3, apoe4, sex, allele, *tmp_res])


columns = [
    "Multiway",
    "APOE2",
    "APOE3",
    "APOE4",
    "Sex",
    "Allele",
] + volumes.structure.to_list()

df = pd.DataFrame(res, columns=columns)

df["K"] = (
    (df[["APOE2", "APOE3", "APOE4"]].sum(axis=1)) * (df["Sex"] + 1) * (df["Allele"] + 1)
)

df.sort_values(by="K", inplace=True, ascending=False)

i_new = []
for idx, row in df.iterrows():
    index = ""
    for col in ["K", "Multiway", "APOE2", "APOE3", "APOE4", "Sex", "Allele"]:
        if col == "K":
            index += str(row[col]) + " | "
        else:
            if row[col]:
                index += "X | "
            else:
                index += "  | "
    i_new.append(index)

df.index = i_new

pvalues = df.iloc[:, 6:-1].copy()

sig, corrected_pvals, b, alpha = multipletests(
    pvalues.values.ravel(), method="bonferroni"
)

pvalues.iloc[:, :] = corrected_pvals.reshape(16, -1)
mask = np.select([pvalues < 0.05, pvalues >= 0.05], ["X", ""], default=pvalues)


# %%
node_labels = pd.read_excel("../data/raw/CHASSSYMM3AtlasLegends.xlsx")[:-1]

left_nodes = (node_labels["Hemisphere"] == "Left").values
right_nodes = (node_labels["Hemisphere"] == "Right").values

nodes_to_choose = ((pvalues <= 0.05).sum() >= 1).values
pvalues.columns = node_labels["Abbreviation"].values

left_idx = nodes_to_choose & left_nodes
right_idx = nodes_to_choose & right_nodes

# %%

cbar_ticks = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 1]

fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
sns.heatmap(
    pvalues.transform("log10").iloc[:, left_idx],
    ax=ax,
    annot=mask[:, left_idx],
    fmt="",
    square=False,
    linewidths=0.5,
    center=np.log10(0.05),
    cbar_kws={"ticks": np.log10(cbar_ticks)},
    cmap="RdBu",
    # xticklabels=False,
)

ax.collections[0].colorbar.set_label("pvalue (log scale, bonferroni-adjusted)")
ax.collections[0].colorbar.set_ticklabels(cbar_ticks)

ax.set_yticklabels(ax.get_yticklabels(), ha="right", fontdict={"family": "monospace"})
ax.set_xlabel("Brain Regions")
ax.set_title("Left Hemisphere")

rot = 30


xposition = -0.335
gap = 0.045
for i, name in enumerate(["K", "Multiway", "APOE2", "APOE3", "APOE4", "Sex", "Allele"]):
    ax.text(
        xposition + (i * gap),
        1.02,
        name,
        ha="left",
        va="bottom",
        rotation=rot,
        transform=ax.transAxes,
    )

fig.savefig(
    "../results/figures/control_diet_left_hemisphere.png", bbox_inches="tight", dpi=300
)

fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
sns.heatmap(
    pvalues.transform("log10").iloc[:, right_idx],
    ax=ax,
    annot=mask[:, right_idx],
    fmt="",
    square=False,
    linewidths=0.5,
    center=np.log10(0.05),
    cbar_kws={"ticks": np.log10(cbar_ticks)},
    cmap="RdBu",
    # xticklabels=False,
)

ax.collections[0].colorbar.set_label("pvalue (log scale, bonferroni-adjusted)")
ax.collections[0].colorbar.set_ticklabels(cbar_ticks)

ax.set_yticklabels(ax.get_yticklabels(), ha="right", fontdict={"family": "monospace"})
ax.set_xlabel("Brain Regions")
ax.set_title("Right Hemisphere")

xposition = -0.335
gap = 0.045
for i, name in enumerate(["K", "Multiway", "APOE2", "APOE3", "APOE4", "Sex", "Allele"]):
    ax.text(
        xposition + (i * gap),
        1.02,
        name,
        ha="left",
        va="bottom",
        rotation=rot,
        transform=ax.transAxes,
    )

fig.savefig(
    "../results/figures/control_diet_right_hemisphere.png", bbox_inches="tight", dpi=300
)

# %%
groups = [
    ["Sex_Female", "Sex_Male", "Diet_Control", "Diet_HFD", "Allele_HN"],
    ["Sex_Female", "Sex_Male", "Allele_HN"],
    ["Diet_Control", "Diet_HFD", "Allele_HN"],
    ["Allele_HN"],
]

genotype_groups = []
for i in [3, 2]:
    genotype_groups += list(
        combinations(["Genotype_APOE22", "Genotype_APOE33", "Genotype_APOE44"], i)
    )
genotype_groups = [list(i) for i in genotype_groups]

TEST_LIST = [i + j for i in genotype_groups for j in groups]

# %%

res = []

for test in TEST_LIST:
    lookup_codes = [lookup[i] for i in test]
    test_multiway_table = multiway_table[:, lookup_codes]
    maximum = np.max(test_multiway_table.sum(axis=1))

    idx = test_multiway_table.sum(axis=1) == maximum

    x = vols[idx]
    y = test_multiway_table[idx]

    tmp_res = []
    for i in range(x.shape[1]):
        stat, pval = Dcorr().test(x[:, i], y)
        tmp_res.append(pval)

    apoe2 = True if "Genotype_APOE22" in test else False
    apoe3 = True if "Genotype_APOE33" in test else False
    apoe4 = True if "Genotype_APOE44" in test else False
    sex = True if "Sex_Female" in test else False
    diet = True if "Diet_HFD" in test else False
    multiway = True if sex or diet else False

    res.append([multiway, apoe2, apoe3, apoe4, sex, diet, *tmp_res])


columns = [
    "Multiway",
    "APOE2",
    "APOE3",
    "APOE4",
    "Sex",
    "Diet",
] + volumes.structure.to_list()

df = pd.DataFrame(res, columns=columns)

df["K"] = (
    (df[["APOE2", "APOE3", "APOE4"]].sum(axis=1)) * (df["Sex"] + 1) * (df["Diet"] + 1)
)

df.sort_values(by="K", inplace=True, ascending=False)

i_new = []
for idx, row in df.iterrows():
    index = ""
    for col in ["K", "Multiway", "APOE2", "APOE3", "APOE4", "Sex", "Diet"]:
        if col == "K":
            index += str(row[col]) + " | "
        else:
            if row[col]:
                index += "X | "
            else:
                index += "  | "
    i_new.append(index)

df.index = i_new

pvalues = df.iloc[:, 6:-1].copy()

sig, corrected_pvals, b, alpha = multipletests(
    pvalues.values.ravel(), method="bonferroni"
)

pvalues.iloc[:, :] = corrected_pvals.reshape(16, -1)
mask = np.select([pvalues < 0.05, pvalues >= 0.05], ["X", ""], default=pvalues)

node_labels = pd.read_excel("../data/raw/CHASSSYMM3AtlasLegends.xlsx")[:-1]

left_nodes = (node_labels["Hemisphere"] == "Left").values
right_nodes = (node_labels["Hemisphere"] == "Right").values

nodes_to_choose = ((pvalues <= 0.05).sum() >= 1).values
pvalues.columns = node_labels["Abbreviation"].values

left_idx = nodes_to_choose & left_nodes
right_idx = nodes_to_choose & right_nodes

# %%

fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
sns.heatmap(
    pvalues.transform("log10").iloc[:, left_idx],
    ax=ax,
    annot=mask[:, left_idx],
    fmt="",
    square=False,
    linewidths=0.5,
    center=np.log10(0.05),
    cbar_kws={"ticks": np.log10(cbar_ticks)},
    cmap="RdBu",
    # xticklabels=False,
)

ax.collections[0].colorbar.set_label("pvalue (log scale, bonferroni-adjusted)")
ax.collections[0].colorbar.set_ticklabels(cbar_ticks)

ax.set_yticklabels(ax.get_yticklabels(), ha="right", fontdict={"family": "monospace"})
ax.set_xlabel("Brain Regions")
ax.set_title("Left Hemisphere")

xposition = -0.335
gap = 0.045
for i, name in enumerate(["K", "Multiway", "APOE2", "APOE3", "APOE4", "Sex", "Diet"]):
    ax.text(
        xposition + (i * gap),
        1.02,
        name,
        ha="left",
        va="bottom",
        rotation=rot,
        transform=ax.transAxes,
    )

fig.savefig(
    "../results/figures/hfd_diet_left_hemisphere.png", bbox_inches="tight", dpi=300
)

fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
sns.heatmap(
    pvalues.transform("log10").iloc[:, right_idx],
    ax=ax,
    annot=mask[:, right_idx],
    fmt="",
    square=False,
    linewidths=0.5,
    center=np.log10(0.05),
    cbar_kws={"ticks": np.log10(cbar_ticks)},
    cmap="RdBu",
    # xticklabels=False,
)

ax.collections[0].colorbar.set_label("pvalue (log scale, bonferroni-adjusted)")
ax.collections[0].colorbar.set_ticklabels(cbar_ticks)

ax.set_yticklabels(ax.get_yticklabels(), ha="right", fontdict={"family": "monospace"})
ax.set_xlabel("Brain Regions")
ax.set_title("Right Hemisphere")

xposition = -0.335
gap = 0.045
for i, name in enumerate(["K", "Multiway", "APOE2", "APOE3", "APOE4", "Sex", "Diet"]):
    ax.text(
        xposition + (i * gap),
        1.02,
        name,
        ha="left",
        va="bottom",
        rotation=rot,
        transform=ax.transAxes,
    )

fig.savefig(
    "../results/figures/hfd_diet_right_hemisphere.png", bbox_inches="tight", dpi=300
)
# %%
