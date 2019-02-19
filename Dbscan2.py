#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 17:30:34 2017

@author: ubuntu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

#Importing the dataset
dataset = pd.read_csv('dataset.csv')

#using the lat and long columns
X = dataset.iloc[:,[1,2]].values
X = StandardScaler().fit_transform(X)

#viualising data
plt.scatter(X[:,0],X[:,1])
#importing DBSCAN
from sklearn.cluster import DBSCAN

#applying DBSCAN
db = DBSCAN(eps=.206, min_samples=6, metric='haversine', algorithm='ball_tree')
db.fit(X) #fir DBSCAN to datatset

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

#Labelling clusters
cluster_labels = db.labels_ #to tell to which cluster data belongs

#to find the number of clusters
num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)


#visualising clusters

unique_labels = set(cluster_labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (cluster_labels == k)

    cluster = X[class_member_mask & core_samples_mask]
    plt.plot(cluster[:, 0], cluster[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

    cluster = X[class_member_mask & ~core_samples_mask]
    plt.plot(cluster[:, 0], cluster[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=1)

plt.title('Estimated number of clusters: %d' % num_clusters)
plt.show()
