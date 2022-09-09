from os.path import dirname, join
from pathlib import Path

import numpy as np
import pandas as pd
from graspologic.utils import symmetrize


DATA_DIR = Path(Path(__file__).absolute().parents[3]) / "data"


def _check_data():
    if not DATA_DIR.exists():
        msg = "Where is the data?"
        raise ValueError(msg)

    key = pd.read_csv(DATA_DIR / "/processed/key.csv")
    genotypes = ["APOE22", "APOE33", "APOE44"]

    mask = key["Genotype"].isin(genotypes)  # for filtering out invalid genotypes
    key = key.loc[mask]

    return key, mask


def load_volume():
    """Loads brain volumes from the 29 mice. There are total of 11 APOE2, 8 APOE3, and 10 APOE4 mice.

    Returns:
        out: 2d array
        labels: 1d array
    """
    key, mask = _check_data()

    volumes = pd.read_csv(DATA_DIR / "/processed/mouses-volumes.csv")
    volumes = volumes[mask]

    out = volumes.to_numpy()

    return out, key["Genotype"]


def load_volume_correlation():
    """Loads the correlation matrices per genotype

    Returns:
        out: 3d array, shape (genotype, num_regions, num_regions)
    """
    data, labels = load_volume()

    correlations = []

    genotypes = np.unique(labels)
    for genotype in genotypes:
        pass

    return 1


def load_fa():
    """_summary_

    Returns:
        _type_: _description_
    """
    key, mask = _check_data()

    return 1


def load_fa_correlation():
    """_summary_

    Returns:
        _type_: _description_
    """
    return 1
