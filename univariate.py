import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps

def histogram(X,axis,bins=30):
    heights,b = np.histogram(X,bins=bins,density=True)
    x = b[:-1]
    w = len(heights)/b[-1]
    axis.bar(x,heights,width=1,color='b')
    axis.set_title('Histogram of Data')

def cumulative_histogram(X,axis,bins=30):
    info = calculate_descriptive_stats(X)
    res = sps.cumfreq(X,numbins=30)
    x = res.lowerlimit + np.linspace(0, res.binsize*res.cumcount.size, res.cumcount.size)
    axis.bar(x, res.cumcount/np.max(res.cumcount), width=res.binsize,color='b',label="Count: %d\nMin: %.4f\nMax: %.4f\nMean: %.4f\nStd: %.4f\nSkew: %.4f\nKurt: %.4f"%info)
    axis.set_title('Cumulative Histogram of Data')
    axis.set_xlim([x.min(),x.max()])
    axis.legend(loc='lower right')

def calculate_descriptive_stats(X):
    info = sps.describe(X)
    mean = info.mean
    std = np.sqrt(info.variance)
    skew = info.skewness
    kurtosis = info.kurtosis
    count = info.nobs
    minimum = info.minmax[0]
    maximum = info.minmax[1]
    return count, minimum, maximum, mean, std, skew, kurtosis

def plot_stats(ax,count,minimum,maximum,mean,std,skew,kurtosis):
    ax.text(0,-10,"Count: %d\nMin: %.4f\nMax: %.4f\nMean: %.4f\nStd: %.4f\nSkew: %.4f\nKurt: %.4f"%(count,minimum,maximum,mean,std,skew,kurtosis))
    ax.set_ylim(-12,-1)
    ax.set_xlim(-1,2)
    ax.axis('off')

def overview(X,bins=30):
    fig,ax = plt.subplots(1,2,figsize=(12,6))
    count,minimum,maximum,mean,std,skew,kurtosis = calculate_descriptive_stats(X)
    histogram(X,ax[0])
    cumulative_histogram(X,ax[1])
    #plot_stats(ax[2],count,minimum,maximum,mean,std,skew,kurtosis)
    plt.show()



