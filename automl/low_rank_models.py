import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from scipy.linalg import qr

def approx_rank(A, threshold=0.03):
    singularValues = np.linalg.svd(A, compute_uv=False)
    rank = singularValues[singularValues >= threshold*singularValues[0]]
    return rank.size

def pivoted_qr(A):
    return qr(A, pivoting=True)[2]

def pca(A, threshold=0.03):
    rank = approx_rank(A, threshold)
    col_stdev = np.std(A, axis=0)
    U,s,Vt = svds(A, k=rank)
    Sigma_sqrt = np.diag(np.sqrt(s))
    X = np.matrix.transpose(np.dot(U, Sigma_sqrt))
    Y = np.dot(np.dot(Sigma_sqrt, Vt), np.diag(col_stdev))
    return X,Y,Vt

#change to nonnegative matrix factorization

def low_rank_approximation(A, a, known_indices, threshold=0.03):
      
    X,Y,_ = pca(A, threshold=threshold)
    # find x using matrix division using known portion of a, corresponding columns of A
    x = np.matrix.transpose(np.linalg.lstsq(np.matrix.transpose(Y[:,known_indices]), np.matrix.transpose(a[:,known_indices]))[0])
  
    # approximate full a as x*Y
    estimatedRow = np.dot(x,Y)
    
    return estimatedRow
