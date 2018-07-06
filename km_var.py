from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def kmeans_explained_variance(k_min, k_max, data):
	# Data is an NxM matrix consisting of N observations and M features
	assert k_max > k_min
	
	# Calculate centroids for each model
	k_means = range(k_min, k_max+1)
	k_means_var = [KMeans(n_clusters=k).fit(data) for k in k_means]
	centroids = [km.cluster_centers_ for km in k_means_var]
	
	# Assign observations in data to closest centroid
	k_euclid = [cdist(df, cent, 'euclidean') for cent in centroids]
	dist = [np.min(ke, axis=1) for ke in k_euclid]
	
	# Within-cluster sum of squares
	wcss = [np.sum(d**2) for d in dist]
	
	# Total sum of squares
	tss = np.sum(pdist(data)**2) / data.shape[0]
	
	# Between-cluster sum of squares
	bcss = tss - wcss

	return (k_means, bcss / tss)
