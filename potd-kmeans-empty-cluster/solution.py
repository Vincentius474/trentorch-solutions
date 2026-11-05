import numpy as np


def kmeans_empty_cluster_update(
    points: np.ndarray,
    assignments: np.ndarray,
    old_centroids: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    points: array of shape (n, d).
    assignments: array of shape (n,), cluster id in [0, k) for each point.
    old_centroids: array of shape (k, d), the previous centroids.
    k: number of clusters.

    Returns an array of shape (k, d): for each cluster, the mean of its
    assigned points, or the corresponding old centroid unchanged if the
    cluster has no assigned points.
    """
    # TODO: group points by assignment, average each nonempty group,
    # and fall back to old_centroids for empty groups.

    new_centroids = old_centroids.copy()

    for cluster_id in range(k):
        # Select points assigned to the current cluster.
        cluster_points = points[assignments == cluster_id]

        # Update only if the cluster contains points.
        if len(cluster_points) > 0:
            new_centroids[cluster_id] = np.mean(
                cluster_points,
                axis=0
            )

    return new_centroids
