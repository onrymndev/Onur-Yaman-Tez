import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE

# -------------------------------------
# Weighting Strategies
# -------------------------------------

def inverse_distance_weight(distance):
    return 1 / (distance + 1e-5)

def gaussian_weight(distance, sigma=1.0):
    return np.exp(- (distance ** 2) / (2 * sigma ** 2))

def exponential_weight(distance, alpha=1.0):
    return np.exp(-alpha * distance)

def rank_weight(rank):
    return 1 / (rank + 1)

def adaptive_power_weight(distance, beta=1.0):
    return 1 / (distance ** beta + 1e-5)

# -------------------------------------
# Weighted Voting with Strategy Selection
# -------------------------------------

def weighted_majority_voting(neighbors_labels, neighbors_distances, strategy, **kwargs):
    if strategy == "inverse":
        weights = inverse_distance_weight(neighbors_distances)
    elif strategy == "gaussian":
        weights = gaussian_weight(neighbors_distances, sigma=kwargs.get('sigma', 1.0))
    elif strategy == "exponential":
        weights = exponential_weight(neighbors_distances, alpha=kwargs.get('alpha', 1.0))
    elif strategy == "rank":
        weights = np.array([rank_weight(rank) for rank in range(len(neighbors_distances))])
    elif strategy == "adaptive":
        weights = adaptive_power_weight(neighbors_distances, beta=kwargs.get('beta', 1.0))
    else:
        raise ValueError("Unknown weighting strategy!")

    label_weights = {}
    for label, weight in zip(neighbors_labels, weights):
        label_weights[label] = label_weights.get(label, 0) + weight

    return max(label_weights, key=label_weights.get)

# -------------------------------------
# ENN with Weighted Voting
# -------------------------------------

def enn_with_weighted_voting(X, y, strategy, n_neighbors=3, **kwargs):
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    keep_indices = []

    for idx, (point, label) in enumerate(zip(X, y)):
        neighbors_distances, neighbors = nn.kneighbors([point])
        neighbors_distances = neighbors_distances.flatten()[1:]
        neighbors_labels = y[neighbors.flatten()[1:]]

        majority_label = weighted_majority_voting(neighbors_labels, neighbors_distances, strategy, **kwargs)

        if label == majority_label:
            keep_indices.append(idx)

    return X[keep_indices], y[keep_indices]

# -------------------------------------
# SMOTEENN - Full Pipeline
# -------------------------------------

def smoteenn_with_weighted_voting(X, y, strategy, n_neighbors_smote=5, n_neighbors_enn=5, random_state=11, **kwargs):
    X = X.values if hasattr(X, 'values') else X
    y = y.values if hasattr(y, 'values') else y

    class_counts = Counter(y)
    minority_class = min(class_counts, key=class_counts.get)

    sm = SMOTE(sampling_strategy='auto', k_neighbors=n_neighbors_smote, random_state=random_state)
    X_resampled, y_resampled = sm.fit_resample(X, y)

    X_final, y_final = enn_with_weighted_voting(X_resampled, y_resampled, strategy, n_neighbors=n_neighbors_enn, **kwargs)

    print("Original class distribution:", class_counts)
    print("After SMOTE:", Counter(y_resampled))
    print("After distance-weighted ENN:", Counter(y_final))

    return X_final, y_final
