from typing import Callable
import numpy as np
from scipy.cluster.vq import vq, kmeans2
from scipy.spatial.distance import cdist


class ProductQuantization:
    def __init__(self, K: int, M: int) -> None:
        """
        Args:
            K (int): Number of clusters dimensions in a sub-partition
            M (int): Number of space sub-partitions
        """
        self.M = M  
        self.K = K  

        def distance_l2(centroids, sub_query):
            # Equvalent to -> cdist([sub_query], self.centroids[m], "sqeuclidean")[0]
            return np.sum(np.power(np.abs(centroids - sub_query), 2), axis=1)

        def distance_dot(centroids, sub_query):
            return (centroids * sub_query).sum(axis=-1)

        self.distances = {"l2": distance_l2, "dot": distance_dot}

    def fit(self, X: np.array, iter: int, seed: int = 123):
        """For each vector in X calculates subspaces centroids using kmeans

        Args:
            X (np.array): 2-D matrix
            iter (int): Number iterations to run kmeans on each M subspace
            seed (int): numpy seed
        """
        dim_rows, dim_col = X.shape

        if dim_col % self.M != 0:
            raise ValueError(
                "X Matrix dimention not divisible by M, not possible to subset X matrix"
            )

        self.delta = int(dim_col / self.M)

        if dim_rows < self.K:
            raise ValueError(f"There are more dimensions K than vector dimensions")

        np.random.seed(seed)
        self.centroids = np.zeros((self.M, self.K, self.delta), dtype=np.float32)
        for m in range(self.M):
            sub_vector = X[:, (self.delta * m) : (self.delta * (m + 1))]
            self.centroids[m], _ = kmeans2(sub_vector, self.K, iter, minit="points")

    def encode(self, X: np.array):
        """Creates X_encoded which has in each column (subspace) the nearest centroid for each instance

        Args:
            X (np.array): Matrix (instances, features)
        """
        self.X_encoded = np.empty((X.shape[0], self.M), np.int8)
        for m in range(self.M):
            X_subspace = X[:, (self.delta * m) : (self.delta * (m + 1))]
            self.X_encoded[:, m], _ = vq(X_subspace, self.centroids[m])

    def fit_encode(self, X: np.array, iter: int):
        self.fit(X, iter)
        self.encode(X)

    def decode(self):
        instances, _ = self.X_encoded.shape
        self.X_decoded = np.empty((instances, self.delta * self.M), dtype=np.float32)
        for m in self.M:
            instances_encoded_values = self.X_encoded[:, m]
            self.X_decoded[:, self.delta * m : self.delta * (m + 1)] = self.centroids[
                m
            ][instances_encoded_values, :]

    def transform(self, query: np.array, queries_distances: Callable, distance: str)-> np.array:
        """Calculates query distances from centroids for each subsapce

        Args:
            query (np.array): Query (instances, )
            calculate_distance (Callable): Function to calculate distance
            distance (str): Type of distance to calculate

        Raises:
            TypeError: Query type not np.float32
            ValueError: Query is not a vector
            TypeError: Query shape is not equal to delta * M

        Returns:
            np.array: Distance from query to X
        """

        if query.dtype != np.float32:
            raise TypeError("Query need to be np.float32")

        if query.ndim != 1:
            raise ValueError("Query needs to be a vector with dimension equal to one")

        if query.shape[0] != self.delta * self.M:
            raise TypeError(
                f"Query dimension does not match with delta={self.delta} * M={self.M}"
            )

        self.query_centers_distance_table = np.empty((self.M, self.K), dtype=np.float32)

        for m in range(self.M):
            sub_query = query[self.delta * m : self.delta * (m + 1)]
            self.query_centers_distance_table[m, :] = self.distances[distance](
                self.centroids[m], sub_query
            )

        distances = queries_distances(self)
        return distances


def queries_asimetric_distance(product_quantization: ProductQuantization) -> np.array:
    """Queries distances from query_centers_distance_table taking into account X_encoded

    Args:
        product_quantization (ProductQuantization): ProductQuantization instance

    Returns:
        np.array: distances (instance, )
    """
    distance = np.sum(
        product_quantization.query_centers_distance_table[
            np.arange(product_quantization.M), product_quantization.X_encoded
        ],
        axis=1,
    )
    return distance
