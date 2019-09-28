import numpy as np

def PCA(X,dims=0,transform=True):
    if transform:
        means = np.mean(X,axis=1).reshape(X.shape[0],1)
        X = np.subtract(X,means)

    U,S,_ = np.linalg.svd(X)
    
    if dims > 0:
        return U[:dims,:], S[:dims]
    else:
        return U, S

def skree_plot(S,axis):
    axis.plot(S)
    axis.set_ylabel(r'$\lambda$')
    axis.set_title('Skree plot of PCs')

def cumulative_variance_plot(S,axis):
    total = np.sum(S)
    axis.plot(np.cumsum(S)/total)
    axis.set_ylabel('Fraction of variance explained')
    axis.set_xlabel('Component')
    axis.set_title('Variance explained using up to nth PC')

def convert_to_graph(X,affinity_measure='euclidean',epsilon=1):
    _,n = X.shape
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if j > i:
                p1 = X[:,i]
                p2 = X[:,j]
                distance = np.linalg.norm(np.subtract(p2,p1))
                A[i,j] = np.exp(-0.5*np.square(distance)/epsilon)
    A = A + A.T
    deg = np.sum(A,axis=1).reshape(n,1)
    A = np.divide(A,2*deg)
    d = np.diag(np.ones(n)*0.5,0)
    return A + d

def _diffusion_map(A):
    U,S,_ = np.linalg.svd(A)
    U = U[:,1:]
    return U

def diffusion_map(X,dims=0,affinity_measure='euclidean'):
    A = convert_to_graph(X,affinity_measure=affinity_measure)
    dm = _diffusion_map(A)
    if dims > 0:
        return dm[:dims,:]
    else:
        return dm

def embedding_2D(X,transform=True,method='PCA',affinity_measure='euclidean'):
    if method == 'PCA':
        X_,_ = PCA(X,dims=2)
        return X_[0,:],X_[1,:]
    elif method == 'DiffMap':
        X_ = diffusion_map(X,dims=2,affinity_measure=affinity_measure)
        return X_[0,:],X_[1,:]


