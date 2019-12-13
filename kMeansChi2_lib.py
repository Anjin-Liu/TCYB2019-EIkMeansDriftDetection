# -*- coding: utf-8 -*-
"""
Created on Fri May 31 17:23:04 2019

@author: Anjin Liu (anjin.liu@uts.edu.au)

Experiment data set for paper submitted to TCYB2019,
Title: Concept Drift Detection via Equal Intensity k-means Space Partitioning
"""

from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
class kMeansChi2():
    
    def __init__(self, k):
        
        self.kmeans = None
        self.k = k
        self.lambdas = None
    
    def buildkMeans(self, data_train, output_path=None):
        
        k = self.k
        X = data_train
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        C_idx = kmeans.labels_
        
        if output_path is not None:
            self.output_partition_result(X, C_idx, output_path)
    
        #===========#
        # plot test #
        #===========#
        for i in range(self.k):
            idx = np.where(C_idx == i)
            plt.scatter(data_train[idx[0], 0], data_train[idx[0], 1])
        plt.show()
        
        lambdas = np.zeros(k)
        k_list, unique_count = np.unique(C_idx, return_counts=True)
        lambdas[k_list] = unique_count
        self.kmeans = kmeans
        self.lambdas = lambdas
        
    def drift_detection(self, data_test, alpha):
        
        kmeans = self.kmeans
        lambdas = self.lambdas
        k = self.k
        
        C_idx = kmeans.predict(data_test)
        observations = np.zeros(k)
        k_list, unique_count = np.unique(C_idx, return_counts=True)
        observations[k_list] = unique_count
        
        h = 0
        contingency_table = np.array([lambdas, observations])
        #print(obs)
        chi2, p, dof, ex = chi2_contingency(contingency_table)
        if p < alpha:
            h = 1
        return h
    
    def output_partition_result(self, X, y, output_path):
        
        output_data = np.hstack([X, y.reshape(-1,1)])
        np.savetxt(output_path, output_data, delimiter=",")
        
        
        
        
        
        