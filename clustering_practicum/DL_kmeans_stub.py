__author__ = 'you'

import numpy as np


class KMeans(object):

    def __init__(self, n_clusters, max_iterations=50, n_init=10):
        self.n_clusters = n_clusters
        self.iterations = max_iterations
        self.n_init = n_init
        self.cluster_centers_ = None
        self.dispersion = None

    def fit(self, X):

        def intra_cluster_dist(n):
            return np.linalg.norm(X[clusters == n] - self.cluster_centers_[n])**2

        def dispersion():
            return sum(map(intra_cluster_dist,
                           range(self.n_clusters)))

        for _ in range(self.n_init):
            lowest_dispersion = np.inf

            clusters = self.run(X)
            Wk = dispersion()

            if Wk < lowest_dispersion:
                lowest_dispersion = Wk
                best_clusters = clusters
                best_centroids = self.cluster_centers_

        self.cluster_centers_ = best_centroids
        self.dispersion = lowest_dispersion
        return best_clusters

    def run(self, X):

        def distance_from_centroid(c):
            return np.linalg.norm(c - X, axis=1)

        def compute_new_centroid(n):
            return np.mean(X[clusters == n], axis=0)

        # init
        self.cluster_centers_ = X[np.random.randint(X.shape[0],
                                                    size=self.n_clusters)]

        for _ in range(self.iterations):

            distance_matrix = np.apply_along_axis(distance_from_centroid,
                                                  axis=1,
                                                  arr=self.cluster_centers_)  # dim(n_clusters, X.shape[0])

            clusters = distance_matrix.argmin(axis=0)  # dim(X.shape[0])

            new_centroids = np.array(map(compute_new_centroid,
                                         range(self.n_clusters)))  # dim(n_clusters, X.shape[1])

            if (new_centroids == self.cluster_centers_).all():
                break

            self.cluster_centers_ = new_centroids

        return clusters


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    n_samples = 500
    np.random.seed(42)
    from sklearn import cluster, datasets
    from sklearn.neighbors import kneighbors_graph
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans as sKMeans

    X, y = datasets.make_blobs(n_samples=500, cluster_std=1.0, random_state=3)
    X = StandardScaler().fit_transform(X)
    X.shape
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    kmeans = KMeans(3)
    kmeans.fit(X)
    y_pred = kmeans.predict(X)
    plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)
    plt.show()
