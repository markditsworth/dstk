import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import seaborn as sns
from marchenko_pastur import *
from embedding import *

def mutual_information(x, y, bins=30):
    c_xy = np.histogram2d(x, y, bins)[0]
    g, p, dof, expected = sps.chi2_contingency(c_xy, lambda_="log-likelihood")
    mi = 0.5 * g / c_xy.sum()
    return mi

def mutual_information_matrix(X,bins=30):
    p,n = X.shape
    mi_matrix = np.zeros((p,p))
    for i in range(p):
        for j in range(p):
            mi_matrix[i,j] = mutual_information(X[i,:],X[j,:],bins=bins)
    return mi_matrix

def plot_mi_matrix(X,ax,bins=30):
    mi_matrix = mutual_information_matrix(X,bins=bins)
    sns.heatmap(mi_matrix,ax=ax,cmap='inferno')
    ax.set_title('Mutual Information Heatmap')

def covariance_matrix(X,transform=True):
    if transform:
        X_ = transform_to_zero_mean_and_unit_std(X)
    else:
        X_ = X.copy()

    C = np.dot(X_,X_.T) / X.shape[1]
    return C

def plot_cov_matrix(X,axis,transform=True):
    C = covariance_matrix(X,transform=transform)
    sns.heatmap(C,ax=axis,cmap='inferno')
    axis.set_title('Covariance Heatmap')

def overview(X,skip=[]):
    fig,ax = plt.subplots(2,2)
    if 'marchenko-pastur' not in skip:
        marchenko_pastur_comparison(X,ax[0,0])
    if 'skree' not in skip:
        _,S = PCA(X)
        skree_plot(S,ax[0,1])
    if 'covariance' not in skip:
        plot_cov_matrix(X,ax[1,0])
    if 'mutual-information' not in skip:
        plot_mi_matrix(X,ax[1,1])
    plt.show()


