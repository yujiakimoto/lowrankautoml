import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import qr


def approx_rank(A, threshold=0.03):
    s = np.linalg.svd(A, compute_uv=False)
    rank = s[s >= threshold*s[0]]
    return rank.size


def pivoted_qr(A):
    return qr(A, pivoting=True)[2]


def pca(A, threshold=0.03):
    rank = approx_rank(A, threshold)
    col_stdev = np.std(A, axis=0)
    U, s, Vt = svds(A, k=rank)
    sigma_sqrt = np.diag(np.sqrt(s))
    X = np.dot(U, sigma_sqrt).T
    Y = np.dot(np.dot(sigma_sqrt, Vt), np.diag(col_stdev))
    return X, Y, Vt


def low_rank_approximation(A, a, known_indices, threshold=0.03):
      
    X, Y, _ = pca(A, threshold=threshold)
    # find x using matrix division using known portion of a, corresponding columns of A
    x = np.linalg.lstsq(np.matrix.transpose(Y[:, known_indices]), np.matrix.transpose(a[:, known_indices]))[0].T
  
    # approximate full a as x*Y
    estimated_row = np.dot(x, Y)
    
    return estimated_row
