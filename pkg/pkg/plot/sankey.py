import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def count_groups(label_matrix):
    levels = label_matrix.shape[1] - 1
    d = []

    for level in range(levels):
        upper_cluster_ids = np.unique(label_matrix[:, level])

        for upper_cluster_id in upper_cluster_ids:
            lower_cluster_ids, counts = np.unique(
                label_matrix[label_matrix[:, level] == upper_cluster_id][:, level + 1],
                return_counts=True,
            )

            for idx, lower_cluster_id in enumerate(lower_cluster_ids):
                d.append((upper_cluster_id, lower_cluster_id, counts[idx]))

    d = np.array(d)

    source = d[:, 0]
    target = d[:, 1]
    value = d[:, 2]

    return source, target, value


def append_apriori_labels(apriori_labels, cluster_matrix):
    encoder = LabelEncoder()
    apriori_labels_encoded = encoder.fit_transform(apriori_labels)
    apriori_labels_encoded = apriori_labels_encoded.reshape(-1, 1)

    # Increase the original cluster_matrix labels
    cluster_matrix_ = cluster_matrix + np.max(apriori_labels_encoded) + 1

    out = np.hstack([apriori_labels_encoded, cluster_matrix_])

    return out, list(encoder.classes_)
