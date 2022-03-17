#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 17:47:53 2021

@author: choueb
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC

from sklearn import metrics
import numpy as np
#charger le dataset

df=pd.read_csv("COVID19_Dataset.csv")

#Nettoyage des nos données
df.dropna(inplace=True)
#Afficher 10 elements 
print(df.head(10))
#voir les dimensions de nos elements
print(df.shape)

#decrire les elements
print(df.describe())

#visualisation avec seaborn
#sns.pairplot(df)
sns.heatmap(df.corr())
#print(df.iloc[1:25,1:6].head(25))
x=df.iloc[1:21000,1:6].values


print(x.shape)
#data train
x_train=x[1:10000,:]
#data test
x_test=x[10001:20000,]
#dataClass ou les données de la classe
x_train_labels=df.iloc[1:10000,1]
x_test_labels=df.iloc[10001:20000,1]

neigh=KNeighborsClassifier(n_neighbors=6)

fit_data=neigh.fit(x_train,x_train_labels)
x_test_pred=fit_data.predict(x_test)
#print(fit_data)
print(x_train_labels.values)


pca = PCA(n_components = 2)
x_train2= pca.fit_transform(x_train)
print(x_train2)
neigh.fit(x_train2,x_train_labels)
plot_decision_regions(x_train2, x_train_labels.values, clf=neigh, legend=2)
### Adding axes annotations
plt.xlabel('X_label')
plt.ylabel('Y_label')
plt.title('K-nearest neighbor')
plt.show()
