import networkx as nx
import numpy as np
from numpy.linalg import inv


def deltacon_similarity_from_adj(A1: np.ndarray, A2: np.ndarray, epsilon: float = 0.01) -> float:
    """
    Compute DeltaCon similarity between two graphs given by adjacency matrices.
    Parameters:
        A1, A2 : np.ndarray
            Binary (0/1) adjacency matrices of same shape (n x n)
        epsilon : float
            Small constant used in the affinity computation
    Returns:
        similarity : float
            DeltaCon similarity in [0, 1]
    """
    # Check dimensions
    if A1.shape != A2.shape:
        raise ValueError("Adjacency matrices must have the same dimensions")
 
    n = A1.shape[0]
    I = np.eye(n)
 
    # Degree matrices
    D1 = np.diag(np.sum(A1, axis=1))
    D2 = np.diag(np.sum(A2, axis=1))
 
    # Affinity matrices
    S1 = inv(I + epsilon**2 * D1 - epsilon * A1)
    S2 = inv(I + epsilon**2 * D2 - epsilon * A2)
 
    # Rooted Euclidean Distance (DeltaCon core)
    diff = np.sqrt(S1) - np.sqrt(S2)
    d = np.sqrt(np.sum(diff**2))
 
    # Convert distance to similarity
    similarity = 1 / (1 + d)
    return similarity


def Connector(Q):
    D = nx.to_networkx_graph(Q,create_using=nx.DiGraph())
    Isolate_list=list(nx.isolates(D))
    if len(Isolate_list)>0:
        for i in Isolate_list:
            if i==0:
                Q[i+1,i]=0.0001
            else:
                Q[i-1,i]=0.0001
    del D
    return Q

 
def kNN(A,N,k):
    np.fill_diagonal(A,0)
    # print(max(A[:,4]))
    # A=np.where(A > 0.09, 1, 0)
 
    # W.sort(reverse=True)
    B1 = np.zeros((N, N))
    for i in range(N):
        W=sorted(A[i,:],reverse=True)
    #     print( W[k])
        B1[i,:]=np.where(A[i,:] > W[k], 1, 0)
 
    # B=np.multiply(B1,A)
    # print(W[k])
    # print(A[20,1:20])
    # print(B[20,1:20])
 
 
    C1 = np.zeros((N, N))
    for i in range(N):
        W=sorted(A[:,i],reverse=True)
    # print( W[k])
        C1[:,i]=np.where(A[:,i] > W[k], 1, 0)
    # C=np.multiply(C1,A)
    Q1=B1+C1    
    Q2=np.where(Q1 > .9 , 1, 0)
 
    Q=np.multiply(Q2,A)
    # del A
    del B1
    del C1
    del Q1
    del Q2 
    Connector(Q)
 
    return Q