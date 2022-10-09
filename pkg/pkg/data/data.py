from os.path import dirname, join
from pathlib import Path

import numpy as np
import pandas as pd
from graspologic.utils import symmetrize

DATA_DIR = Path(Path(__file__).absolute().parents[3]) / "data"

GENOTYPES = [
    "APOE22",
    "APOE33",
    "APOE44",
]
HEMISPHERES = {
    "L": "Left",
    "R": "Right",
}
SUPER_STRUCTURES = {
    "FB": "Forebrain",
    "HB": "Hindbrain",
    "MB": "Midbrain",
    "VS": "Ventricular system",
    "WM": "White matter",
}
SUB_STRUCTURES = {
    "IS": "Isocortex",
    "PA": "Pallium",
    "DI": "Diencephalon",
    "SP": "Subpallium",
    "MB": "MB",
    "BS": "Brain Stem",
    "PP": "Prepontine",
    "HB": "HB",
    "PO": "Pontine",
    "PM": "Pontomedullar",
    "ME": "Medullary",
    "FB": "FB",
    "VS": "VS",
}


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

    mask = key["Genotype"].isin(GENOTYPES)  # for filtering out invalid genotypes
    key = key.loc[mask]

    labels = key["Genotype"].to_numpy()

    return labels, mask


def load_vertex_df():
    df = pd.read_csv(DATA_DIR / "processed/node_label_dictionary.csv")

    cols = [
        "Structure",
        "Abbreviation",
        "Hemisphere_abbrev",
        "Level_1_abbrev",
        "Subdivision_new",
    ]
    new_cols = ["Structure", "Abbreviation", "Hemisphere", "Level_1", "Level_2"]

    df = df.loc[:, cols]
    df.columns = new_cols

    return df


def load_vertex_metadata():
    """Loads vertex wise labels

    Returns
    -------
    labels : tuple
        Elements are (region names, region hemispheres, region super structures,
        region hemisphere + super structures)
    """
    df = pd.read_csv(DATA_DIR / "processed/node_label_dictionary.csv")

    # Region names
    region_names = df.Abbreviation.to_numpy()

    # Region hemispheres
    region_hemispheres = df.Hemisphere
    to_replace = dict(  # Shorten hemisphere labels
        [
            ("Left", "L"),
            ("Right", "R"),
        ]
    )
    region_hemispheres = region_hemispheres.replace(to_replace).to_numpy()

    # Region structures
    region_super_structures = df.Level_1
    to_replace = dict(  # Relabel Level_1 column
        [
            ("1_forebrain", "FB"),
            ("2_midbrain", "MB"),
            ("3_hindbrain", "HB"),
            ("4_white_matter_tracts", "WM"),
            ("5_ventricular_system", "VS"),
        ]
    )
    region_super_structures = region_super_structures.replace(to_replace).to_numpy()

    # Region hemispheres + structures
    region_hemisphere_structures = np.array(
        [i + j for i, j in zip(region_hemispheres, region_super_structures)]
    )

    return (
        region_names,
        region_hemispheres,
        region_super_structures,
        region_hemisphere_structures,
    )


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

    for genotype in GENOTYPES:
        subset = data[labels == genotype]
        corr_arr = symmetrize(np.corrcoef(subset, rowvar=False))
        correlations.append(corr_arr)

    correlations = np.array(correlations)

    return correlations, GENOTYPES


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

    for genotype in GENOTYPES:
        subset = data[labels == genotype]
        corr_arr = symmetrize(np.corrcoef(subset, rowvar=False))
        correlations.append(corr_arr)

    correlations = np.array(correlations)

    return correlations, GENOTYPES
