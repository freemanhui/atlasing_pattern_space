import numpy as np

def _pairwise_d2(A: np.ndarray) -> np.ndarray:
    s = np.sum(A*A, axis=1, keepdims=True)
    d2 = s + s.T - 2*A@A.T
    np.fill_diagonal(d2, np.inf)
    return d2

def knn_indices(arr: np.ndarray, k: int) -> np.ndarray:
    d2 = _pairwise_d2(arr)
    return np.argsort(d2, axis=1)[:, :k]

def knn_preservation(ref: np.ndarray, test: np.ndarray, k: int = 10) -> float:
    a = knn_indices(ref, k)
    b = knn_indices(test, k)
    j = []
    for i in range(ref.shape[0]):
        A = set(a[i]); B = set(b[i])
        u = len(A|B); inter = len(A&B)
        j.append(inter/u if u else 0.0)
    return float(np.mean(j))

def trustworthiness(X: np.ndarray, X_emb: np.ndarray, k: int = 10) -> float:
    # Trustworthiness without sklearn (following the textbook definition)
    n = X.shape[0]
    # rank neighbors in original space
    dX = _pairwise_d2(X); orderX = np.argsort(dX, axis=1)
    # rank neighbors in embedded space
    dY = _pairwise_d2(X_emb); orderY = np.argsort(dY, axis=1)
    ranksX = [ {nbr: rank for rank, nbr in enumerate(orderX[i])} for i in range(n) ]
    T = 0.0
    for i in range(n):
        U = set(orderY[i][:k]) - set(orderX[i][:k])
        T += sum((ranksX[i].get(u, n) - k) for u in U)
    return 1.0 - (2.0/(n*k*(2*n - 3*k - 1))) * T
