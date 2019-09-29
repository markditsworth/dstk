import numpy as np
import scip.stats as sps
import matplotlib.pyplot as plt
import seaborn as sns
from marchenko_pastur import *

def mutual_information(x, y, bins=30):
    c_xy = np.histogram2d(x, y, bins)[0]
    g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
    mi = 0.5 * g / c_xy.sum()
    return mi

def mutual_information_matrix(X,bins=30):
    p,n = X.shape
    mi_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            mi_matrix[i,j] = mutual_information(X[:,i],X[:,j],bins=bins)
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
    ax.set_title('Covariance Heatmap')
