from typing import Optional, cast

import graspologic as gp
import numpy as np
from graspologic.align import OrthogonalProcrustes
from graspologic.embed import AdjacencySpectralEmbed, OmnibusEmbed, select_dimension
from graspologic.simulations import rdpg
from graspologic.utils import is_symmetric
from joblib import Parallel, delayed


def vertex_position_test(
    A1,
    A2,
    embedding="ase",
    n_components=None,
    test_case="rotation",
    n_bootstraps=500,
    workers=1,
):
    num_components: int
    if n_components is None:
        # get the last elbow from ZG for each and take the maximum
        num_dims1 = select_dimension(A1)[0][-1]
        num_dims2 = select_dimension(A2)[0][-1]
        num_components = max(num_dims1, num_dims2)
    else:
        num_components = n_components

    Xhat, Yhat = _embed(A1, A2, embedding, num_components)

    stat = _difference_norm(Xhat, Yhat, embedding, test_case)

    # Compute null distributions
    null_distribution_1 = Parallel(n_jobs=workers)(
        delayed(_bootstrap)(Xhat, embedding, num_components, n_bootstraps, test_case)
        for _ in range(n_bootstraps)
    )
    null_distribution_1 = np.hstack(null_distribution_1)

    null_distribution_2 = Parallel(n_jobs=workers)(
        delayed(_bootstrap)(Yhat, embedding, num_components, n_bootstraps, test_case)
        for _ in range(n_bootstraps)
    )
    null_distribution_2 = np.hstack(null_distribution_2)

    p_value_1 = (null_distribution_1 >= stat).sum(axis=1) / (n_bootstraps + 1)
    p_value_2 = (null_distribution_2 >= stat).sum(axis=1) / (n_bootstraps + 1)

    p_value = np.max([p_value_1, p_value_2], axis=0)

    return stat, p_value, null_distribution_1, null_distribution_2


def _difference_norm(
    X1: np.ndarray,
    X2: np.ndarray,
    embedding,
    test_case,
):
    if embedding in ["ase"]:
        if test_case == "rotation":
            pass
        elif test_case == "scalar-rotation":
            X1 = X1 / np.linalg.norm(X1, ord="fro")
            X2 = X2 / np.linalg.norm(X2, ord="fro")
        elif test_case == "diagonal-rotation":
            normX1 = np.sum(X1**2, axis=1)
            normX2 = np.sum(X2**2, axis=1)
            normX1[normX1 <= 1e-15] = 1
            normX2[normX2 <= 1e-15] = 1
            X1 = X1 / np.sqrt(normX1[:, None])
            X2 = X2 / np.sqrt(normX2[:, None])
        aligner = OrthogonalProcrustes()
        X1 = aligner.fit_transform(X1, X2)
    elif embedding == "omnibus":
        raise NotImplementedError()

    return np.linalg.norm(
        X1 - X2, axis=1, keepdims=True
    )  # change to return stat per vertex


def _embed(
    A1: np.ndarray,
    A2: np.ndarray,
    embedding,
    n_components: int,
    check_lcc: bool = False,
):
    if embedding == "ase":
        X1_hat = cast(
            np.ndarray,
            AdjacencySpectralEmbed(
                n_components=n_components, check_lcc=check_lcc
            ).fit_transform(A1),
        )
        X2_hat = cast(
            np.ndarray,
            AdjacencySpectralEmbed(
                n_components=n_components, check_lcc=check_lcc
            ).fit_transform(A2),
        )
    elif embedding == "omnibus":
        X_hat_compound = OmnibusEmbed(
            n_components=n_components, check_lcc=check_lcc
        ).fit_transform((A1, A2))
        X1_hat = X_hat_compound[0]
        X2_hat = X_hat_compound[1]
    return (X1_hat, X2_hat)


def _bootstrap(
    X_hat: np.ndarray,
    embedding,
    n_components: int,
    test_case,
    bootstrap_method="default",
    rescale: bool = False,
    loops: bool = False,
):
    if bootstrap_method == "default":
        A1_simulated = rdpg(X_hat, rescale=rescale, loops=loops)
        A2_simulated = rdpg(X_hat, rescale=rescale, loops=loops)
        X1_hat_simulated, X2_hat_simulated = _embed(
            A1_simulated, A2_simulated, embedding, n_components, check_lcc=False
        )
        t_bootstrap = _difference_norm(
            X1_hat_simulated, X2_hat_simulated, embedding, test_case
        )
        return t_bootstrap
    elif bootstrap_method == "test1":
        n = X_hat.shape[0]
        idx = np.random.permutation(n)

        A1_simulated = rdpg(X_hat, rescale=rescale, loops=loops)
        A2_simulated = rdpg(X_hat[idx], rescale=rescale, loops=loops)
        X1_hat_simulated, X2_hat_simulated = _embed(
            A1_simulated, A2_simulated, embedding, n_components, check_lcc=False
        )
        t_bootstrap = _difference_norm(
            X1_hat_simulated, X2_hat_simulated, embedding, test_case
        )
        return t_bootstrap
    else:
        raise ValueError("Invalid bootstrap_method!")
