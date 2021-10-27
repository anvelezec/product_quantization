import timeit
import numpy as np
from scipy.stats import pearsonr as corr
from sklearn.datasets import load_digits

import nanopq
from pq.product_quantization import ProductQuantization, queries_asimetric_distance


def nano_product_quantization(X, Xt, query):
    # Instantiate with M=8 sub-spaces
    pq = nanopq.PQ(M=8)

    # Train codewords
    pq.fit(Xt)

    # Encode to PQ-codes
    X_code = pq.encode(X)  # (10000, 8) with dtype=np.uint8

    # Results: create a distance table online, and compute Asymmetric Distance to each PQ-code
    dists = pq.dtable(query).adist(X_code)  # (10000, )
    return dists


def product_quantization(X, Xt, query, distance):
    # Instantiate with M=8 sub-spaces
    pq = ProductQuantization(K=256, M=8)

    # Train codewords
    pq.fit(Xt, iter=20)

    # Encode to PQ-codes
    pq.encode(X)

    # Results: create a distance table online, and compute Asymmetric Distance to each PQ-code
    dists = pq.transform(query, queries_asimetric_distance, distance)
    return dists


def test_product_quantization_l2():
    N, Nt, D = 10000, 2000, 128

    # 10,000 128-dim vectors to be indexed
    X = np.random.random((N, D)).astype(np.float32)

    # 2,000 128-dim vectors for training
    Xt = np.random.random((Nt, D)).astype(np.float32)

    # a 128-dim query vector
    query = np.random.random((D,)).astype(np.float32)

    assert np.allclose(
        nano_product_quantization(X, Xt, query),
        product_quantization(X, Xt, query, "l2"),
    )


def test_product_quantization_dot():
    # for simplicity, use the sklearn digits dataset; we'll split
    # it into a matrix X and a set of queries Q
    X, _ = load_digits(return_X_y=True)
    nqueries = 20
    X, queries = X[:-nqueries], X[-nqueries:]

    # Instantiate with M=8 sub-spaces
    pq = ProductQuantization(K=256, M=8)
    # Train codewords
    pq.fit(X, iter=20)
    # Encode to PQ-codes
    pq.encode(X)

    correlations = []
    for query in queries:
        dists = pq.transform(
            query.astype(np.float32), queries_asimetric_distance, "dot"
        )

        dot_dist = X @ query
        correlations.append(corr(dists, dot_dist)[0] > 0.98)

    assert all(correlations)
