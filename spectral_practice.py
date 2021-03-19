from sklearn.datasets import make_circles
from sklearn.neighbors import kneighbors_graph
import numpy as np

X = [[1,2], [2,3], [30,33], [32,30], [2,1], [31,31]]


A = kneighbors_graph(X, n_neighbors=2).toarray()

#[[0. 1. 0. 0.]
# [1. 0. 0. 0.]
# [0. 0. 0. 1.]
# [0. 0. 1. 0.]]


# create the graph laplacian
D = np.diag(A.sum(axis=1))
L = D-A
print("L:\n", L)
#[[ 1. -1.  0.  0.]
# [-1.  1.  0.  0.]
# [ 0.  0.  1. -1.]
# [ 0.  0. -1.  1.]]

# find the eigenvalues and eigenvectors
vals, vecs = np.linalg.eig(L)
idx = vals.argsort()[::1]
vals = vals[idx]
vecs = vecs[:,idx]


print("Eigenvalues\n:", vals)
# Vals [2. 0. 2. 0.]
# iVecs
#[[ 0.70710678  0.70710678  0.          0.        ]
# [-0.70710678  0.70710678  0.          0.        ]
# [ 0.          0.          0.70710678  0.70710678]
# [ 0.          0.         -0.70710678  0.70710678]]

# use Fiedler value to fi
clusters = vecs[:,1] < 0
print("Eigenvector: ", vecs[:,1])
print("Clusters: ", clusters)
