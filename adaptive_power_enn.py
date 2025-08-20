from sklearn.neighbors import NearestNeighbors
import numpy as np

def adaptive_power_enn_fast(X, y, beta=0.75, k=3):
    """
    Vectorized Edited Nearest Neighbours (ENN) with adaptive power weighting.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix.
    y : array-like, shape (n_samples,)
        Labels.
    beta : float
        Power decay parameter for weighting distances.
    k : int
        Number of nearest neighbours (excluding the sample itself).

    Returns
    -------
    X_res, y_res : ndarray
        Cleaned feature matrix and labels.
    """
    X = np.array(X)
    y = np.array(y)
    
    # Find nearest neighbors (k + 1 because first neighbor is the sample itself)
    nn = NearestNeighbors(n_neighbors=k+1).fit(X)
    dists, indices = nn.kneighbors(X)
    
    # Remove the self-neighbor
    dists = dists[:, 1:]       # shape: (n_samples, k)
    indices = indices[:, 1:]   # shape: (n_samples, k)
    
    neigh_labels = y[indices]  # shape: (n_samples, k)
    
    # Compute adaptive power weights: w = 1 / (d^beta + eps)
    weights = 1.0 / (np.power(dists, beta) + 1e-8)
    
    # Weighted sum per class
    classes = np.unique(y)
    class_scores = np.zeros((X.shape[0], len(classes)))
    for ci, cls in enumerate(classes):
        mask = (neigh_labels == cls).astype(float)  # 1 if neighbor has class cls
        class_scores[:, ci] = (weights * mask).sum(axis=1)
    
    # Pick class with max weighted score
    maj_class_indices = np.argmax(class_scores, axis=1)
    maj_classes = classes[maj_class_indices]
    
    # Keep only samples where weighted majority == actual label
    keep_mask = (maj_classes == y)
    
    return X[keep_mask], y[keep_mask]
