import numpy as np
from scipy.cluster.vq import vq, kmeans2

x_type = np.array
iter_type = int


class ProductQuantization:
    def __init__(self, K: int, M:int) -> None:
        self.M = M # Number space sub-partitions
        self.K = K # Number of cluster in a sub-partition
        

    def fit(self, X: x_type, iter:iter_type):
        """Subsebs Matrix X and estimates centroids using kmeans

        Args:
            X (x_type): 2-D matrix
            iter (iter_type): Number iterations to run kmeans on each M subspace
        """
        dim_row, dim_col = X.shape

        if dim_col % self.M != 0:
            raise("X Matrix dimention not divisible by M, not possible to subset X matrix")

        self.delta = int(dim_col / self.M)

        if self.delta < self.K:
            raise(f"There are more dimensions K than vector dimensions")

        self.centroids = np.zeros((self.M, self.K, self.delta), dtype=np.float32)
        for m in range(self.M):
            sub_vector = X[:, (self.delta * m):(self.delta * (m + 1))]
            self.centroids[m], _ = kmeans2(sub_vector, self.K, iter)

    def encode(self, X: x_type):
        # instances X M
        self.X_encoded = np.empty((X.shape[0], self.M))
        for m in range(self.M):
            X_subspace = X[:, (self.delta * m):(self.delta * (m + 1))]
            self.X_encoded[:, m], _ = vq(X_subspace, self.centroids[m])

    def fit_encode(self, X, iter):
        self.fit(X, iter)
        self.encode(X)
        
    def decode(self):
        pass

if __name__ == "__main__":
    X = np.random.randn(15,40)
    pq = ProductQuantization(K=4, M=8)
    pq.fit(X, iter=20)
    pq.encode(X)
    pq