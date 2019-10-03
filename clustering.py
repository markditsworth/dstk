import numpy as np
import scipy.linalg as spla
from embedding import convert_to_graph

def get_degree_matrix(A):
    return np.diag(np.sum(A,axis=1),0)

def get_laplacian(A):
    return get_degree_matrix(A) - A

def _spectral_clustering_by_connected_components(L):
    n,_ = L.shape[0]
    u,v = spla.eigh(L)
    number_of_components = 1 + np.where(np.abs(u)<1e-14)[0][-1]
    clusters = np.zeros(n)
    for x in range(number_of_components):
        mask = np.abs(v[:,x]) > 1e-14
        clusters[mask] = x
    shift = number_of_components // 2
    return clusters - shift

# Spectral clustering using the Fiedler vector (second-smallest eigenvector)
#  Relaxing the integer conditions in minimizing the number of edges between partitions
#  leads to the solution of the second-smallest eigenvector. (The eigenvector for the smallest
#  eigenvalue assigns all nodes to the same partition, assuming the graph is connected).
#  
# See www.blog.shriphani.com/2015/04/06/the-smallest-eigenvalues-of-a-graph-laplacian
#
def _spectral_clustering_by_fiedler_vector(L,kernel='sgn',explore=False):
    u,v = spla.eigh(L,eigvals=(0,1))
    assert u[1] > 1e-14, "Multiplicity of 0 eigenvalues is > 1. Multiple connected components exist."

    clusters = v[:,1]
    
    if explore:
        return clusters
    if kernel == 'sgn':
        return np.sign(clusters)
    elif kernel == 'mean':
        return(v[:,1] > np.mean(v[:,1])).astype(int)*2 - 1
    elif kernel == 'median':
        return(v[:,1] > np.median(v[:,1])).astype(int)*2 - 1

def spectral_clustering(X,method='fiedler',affinity_measure='euclidean',epsilon=1,truncate=False,threshold=0.1,kernel='sgn',explore=False):
    A = convert_to_graph(X,affinity_measure=affinity_measure,epsilon=1,truncate=truncate,threshold=threshold)
    L = get_laplacian(A)
    if method == 'fiedler':
        clusters = _spectral_clustering_by_fiedler_vector(L,kernel=kernel,explore=explore)
    elif method == 'components':
        clusters = _spectral_clustering_by_connected_components(L)

    return clusters

# shows histogram of affinity measurements to help tune epsilon and to help set an appropriate threshold if truncating
def explore_graph_formation(X,affinity_measure='euclidean',epsilon=1):
    As = convert_to_graph(X,affinity_measure=affinity_measure,epsilon=epsilon,explore=True)
    fig = plt.figure(figsize=(12,8))
    plt.hist(As,bins=30,color='b')
    plt.xlabel('affinity measurement')
    plt.ylabel('frequency')
    plt.title('Distribution of affinity measurements with epsilon=%.3f'%epsilon)
    plt.show()

# returns histogram of pre-bucketed clusters to aid in deciding which kernel to use
#  if distribution is not centered about 0, mean may be best
#  if distribution is skewed, median may be best
def explore_spectral_clustering(X,affinity_measure='euclidean',epsilon=1,truncate=False,threshold=0.1):
    vec = spectral_clustering(X,method='fiedler',affinity_measure=affinity_measure,epsilon=epsilon,truncate=truncate,threshold=threshold,explore=True)
    fig = plt.figure(figsize=(12,8))
    plt.hist(vec,bins=30,color='b')
    plt.xlabel('raw cluster assignment')
    plt.ylabel('frequency')
    plt.title('Distribution of pre-discretized cluster assignments with epsilon=%.3f and threshold=%.3f'%(epsilon,threshold))
    plt.show()
