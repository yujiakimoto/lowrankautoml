import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from scipy.linalg import qr

def approx_rank(A):
    singularValues = np.linalg.svd(A, compute_uv=False)
    rank = singularValues[singularValues >= 0.03*singularValues[0]]
    return rank.size

def pivoted_qr(A):
    return qr(A, pivoting=True)[2]

def pca(A):
    rank = approx_rank(A)
    col_stdev = np.std(A, axis=0)
    U,s,Vt = svds(A, k=rank)
    Sigma_sqrt = np.diag(np.sqrt(s))
    X = np.matrix.transpose(np.dot(U, Sigma_sqrt))
    Y = np.dot(np.dot(Sigma_sqrt, Vt), np.diag(col_stdev))
    return X,Y,Vt

def low_rank_approximation(A, a, known_indices):
      
    X,Y = pca(A)       
    # find x using matrix division using known portion of a, corresponding columns of A
    x = np.matrix.transpose(np.linalg.lstsq(np.matrix.transpose(Y[:,known_indices]), np.matrix.transpose(a[:,known_indices]))[0])
  
    # approximate full a as x*Y
    estimatedRow = np.dot(x,Y)
    
    return estimatedRow
