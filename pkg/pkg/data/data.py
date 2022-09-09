from os.path import dirname, join
from pathlib import Path

import numpy as np
import pandas as pd
from graspologic.utils import symmetrize


DATA_DIR = Path(Path(__file__).absolute().parents[3]) / "data"


def _check_data():
    """_summary_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if not DATA_DIR.exists():
        msg = "Where is the data?"
        raise ValueError(msg)

    key = pd.read_csv(DATA_DIR / "processed/key.csv")
    genotypes = ["APOE22", "APOE33", "APOE44"]

    mask = key["Genotype"].isin(genotypes)  # for filtering out invalid genotypes
    key = key.loc[mask]

    labels = key["Genotype"].to_numpy()

    return labels, mask


def load_vertex_labels():
    """_summary_

    Returns:
        _type_: _description_
    """

    df = pd.read_csv(DATA_DIR / "processed/mouses-volumes.csv")
    labels = df.columns.to_list()

    return labels


def load_volume():
    """Loads brain volumes from the 29 mice. There are total of 11 APOE2, 8 APOE3, and 10 APOE4 mice.

    Returns:
        out: 2d array
        labels: 1d array
    """
    labels, mask = _check_data()

    volumes = pd.read_csv(DATA_DIR / "processed/mouses-volumes.csv")
    volumes = volumes[mask]

    out = volumes.to_numpy()

    return out, labels


def load_volume_corr():
    """Loads the correlation matrices per genotype

    Returns:
        out: 3d array, shape (genotype, num_regions, num_regions)
    """
    data, labels = load_volume()

    correlations = []

    genotypes = ["APOE22", "APOE33", "APOE44"]
    for genotype in genotypes:
        subset = data[labels == genotype]
        corr_arr = symmetrize(np.corrcoef(subset, rowvar=False))
        correlations.append(corr_arr)

    correlations = np.array(correlations)

    return correlations, genotypes


def load_fa():
    """_summary_

    Returns:
        _type_: _description_
    """
    labels, mask = _check_data()

    volumes = pd.read_csv(DATA_DIR / "processed/mouses-fa.csv")
    volumes = volumes[mask]

    out = volumes.to_numpy()

    return out, labels


def load_fa_corr():
    """_summary_

    Returns:
        _type_: _description_
    """
    data, labels = load_fa()

    correlations = []

    genotypes = ["APOE22", "APOE33", "APOE44"]
    for genotype in genotypes:
        subset = data[labels == genotype]
        corr_arr = symmetrize(np.corrcoef(subset, rowvar=False))
        correlations.append(corr_arr)

    correlations = np.array(correlations)

    return correlations, genotypes
