from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def kmeans_explained_variance(k_min, k_max, data):
	assert k_max > k_min
	k_means = range(k_min, k_max+1)
	k_means_var = [KMeans(n_clusters=k).fit(data) for k in k_means]
	centroids = [km.cluster_centers_ for km in k_means_var]
	k_euclid = [cdist(df, cent, 'euclidean') for cent in centroids]
	dist = [np.min(ke, axis=1) for ke in k_euclid]
	wcss = [np.sum(d**2) for d in dist]
	tss = np.sum(pdist(data)**2) / data.shape[0]
	bcss = tss - wcss

	return (k_means, bcss / tss)
