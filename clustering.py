#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 12:46:08 2021

@author: choueb
"""
import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

df=pd.read_csv("COVID19_Dataset.csv")

print(df.shape)

print(df.head(5))
df.dropna(inplace=True)

categorical_features_idx = [0, 6, 7, 8,9]
df=df.iloc[:,:]

mark_array=df.values

print(mark_array)


kproto = KPrototypes(n_clusters=2, verbose=2, max_iter=20).fit(mark_array, categorical=categorical_features_idx)

# Cluster Centroids
print(kproto.cluster_centroids_)

# Prediction
cluster = kproto.predict(mark_array, categorical=categorical_features_idx)
print(type(cluster))
pca = PCA(n_components = 2)
df= pca.fit_transform(df.iloc[:,1:6])
print(df.shape)

#df['cluster'] = list(cluster)
#print(df.head(10))
#print(df[df['cluster']== 0].head(10))

#filter rows of original data
filtered_label0 = df[cluster == 0]
filtered_label1 = df[cluster == 1]
 
plt.scatter(filtered_label0[:,0] , filtered_label0[:,1] , color = 'red')
plt.scatter(filtered_label1[:,0] , filtered_label1[:,1] , color = 'black')
plt.show()
