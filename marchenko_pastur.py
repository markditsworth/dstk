import numpy as np
import matplotlib.pyplot as plt

def _marchenko_pastur(x,gamma,sigma=1.0):
    y=1/gamma
    largest_eigenval = np.power(sigma*(1 + np.sqrt(1/gamma)),2)
    smallest_eigenval= np.power(sigma*(1 - np.sqrt(1/gamma)),2)
    mp = (1/(2*np.pi*sigma*sigma*x*y))*np.sqrt((largest_eigenval - x)*(x - smallest_eigenval))*(0 if (x>largest_eigenval or x<smallest_eigenval) else 1)
    return mp

def marchenko_pastur(n,p,upper_bound=3,spacing=2000,sigma=1.0):
    x_mp_dist = np.linspace(0,upper_bound,spacing)
    y_mp_dist = [_marchenko_pastur(x,n/p,sigma=sigma) for x in x_mp_dist]

    return x_mp_dist,y_mp_dist

def eigenval_histogram(X,bins=100):
    p,n = X.shape
    S = (1./n) * np.dot(X,X.T)
    u,_ = np.linalg.eig(S)

    hist_heights, hist_bins = np.histogram(u,bins=bins,density=True)
    x = hist_bins[:-1]
    y = hist_heights
    col_widths = hist_bins[-1]/len(x)


    return x,y,col_widths

def transform_to_zero_mean_and_unit_std(X):
    m,n = X.shape
    means = np.mean(X,axis=1).reshape(m,1)
    mean_adjusted_X = np.subtract(X,means)
    stds = np.std(mean_adjusted_X,axis=1).reshape(m,1)
    mean_and_std_adjusted_X = np.divide(mean_adjusted_X,stds)
    return mean_and_std_adjusted_X

def marchenko_pastur_comparison(X,axis,histogram_bins=100,dist_upper_bound=3,dist_spacing=2000,sigma=1.0,transform=True):
    p,n = X.shape
    x_mp, y_mp = marchenko_pastur(n,p,upper_bound=dist_upper_bound,spacing=dist_spacing,sigma=sigma)

    if transform:
        X_ = transform_to_zero_mean_and_unit_std(X)
        x,y,w = eigenval_histogram(X_,bins=histogram_bins)
    else:
        x,y,w = eigenval_histogram(X,bins=histogram_bins)

    axis.bar(x,y,width=w,color='b')
    axis.plot(x_mp,y_mp,color='r')
    axis.set_xlabel(r'$\lambda$')
    axis.set_title('Marchenko-Pastur Distribution Comparison')
