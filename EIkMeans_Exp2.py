# -*- coding: utf-8 -*-
"""
Created on Fri May 31 17:23:04 2019

@author: Anjin Liu (anjin.liu@uts.edu.au)

Experiment data set for paper submitted to TCYB2019,
Title: Concept Drift Detection via Equal Intensity k-means Space Partitioning
"""

import numpy as np
import EIkMeans_lib as eikm
    
def run_test(data_train, W0, W1, cp_inst=None, alpha=0.05, k=25, flag=1):
    
    if cp_inst is None:
        m_test = W0.shape[0]
        cp_inst = eikm.EIkMeans(k)
        cp_inst.build_partition(data_train, m_test)
        results = np.zeros(2)
        results[0] = cp_inst.drift_detection(W0, alpha)
        results[1] = cp_inst.drift_detection(W1, alpha)
        return results, cp_inst
    else:
        if flag == 0:
            cp_inst.fill_lambda(data_train)
        results = np.zeros(2)
        results[0] = cp_inst.drift_detection(W0, alpha)
        results[1] = cp_inst.drift_detection(W1, alpha)
        return results, cp_inst

def exp2_evaluation(r_seed, the_k, datasetNameSet, alpha=0.05):
    
    np.random.seed(r_seed)
    
    N1 = 2000
    N2 = 3000
    N3 = 4000
    N4 = 5000
    
    the_alpha = alpha
    
    dataFolderPath_Main = 'Datasets/Exp2_SynDriftDetection/'
    output = np.zeros([len(datasetNameSet)*4, 2])
    dataset_id = 0
    
    for datasetName in datasetNameSet:
        dataFolderPath = dataFolderPath_Main + datasetName
        data_train = np.loadtxt(dataFolderPath+'data_train.csv', dtype=float, delimiter=',')
        
        data_train_N1 = data_train[:N1,:]
        data_train_N2 = data_train[:N2,:]
        data_train_N3 = data_train[:N3,:]
        data_train_N4 = data_train[:N4,:]
        
        test_num = 250
        
        results_final_N1 = np.zeros(2)
        results_final_N2 = np.zeros(2)
        results_final_N3 = np.zeros(2)
        results_final_N4 = np.zeros(2)

        cp_N1 = None
        cp_N2 = None
        cp_N3 = None
        cp_N4 = None
        
        t=0
        
        W0 = np.loadtxt(dataFolderPath+'data_W0_'+str(t)+'.csv', dtype=float, delimiter=',')
        W1 = np.loadtxt(dataFolderPath+'data_W1_'+str(t)+'.csv', dtype=float, delimiter=',')   
        results_N1, cp_N1 = run_test(data_train_N1, W0, W1, cp_inst=cp_N1, alpha=the_alpha, k=the_k, flag=0)
        results_N2, cp_N2 = run_test(data_train_N2, W0, W1, cp_inst=cp_N1.get_copy(), alpha=the_alpha, k=the_k, flag=0)
        results_N3, cp_N3 = run_test(data_train_N3, W0, W1, cp_inst=cp_N1.get_copy(), alpha=the_alpha, k=the_k, flag=0)
        results_N4, cp_N4 = run_test(data_train_N4, W0, W1, cp_inst=cp_N1.get_copy(), alpha=the_alpha, k=the_k, flag=0)
        results_final_N1 = results_final_N1 + results_N1
        results_final_N2 = results_final_N2 + results_N2
        results_final_N3 = results_final_N3 + results_N3
        results_final_N4 = results_final_N4 + results_N4
        
        for t in range(1, test_num):
            
            W0 = np.loadtxt(dataFolderPath+'data_W0_'+str(t)+'.csv', dtype=float, delimiter=',')
            W1 = np.loadtxt(dataFolderPath+'data_W1_'+str(t)+'.csv', dtype=float, delimiter=',')
            results_N1, cp_N1 = run_test(data_train_N1, W0, W1, cp_inst=cp_N1, alpha=the_alpha, k=the_k, flag=1)
            results_N2, cp_N2 = run_test(data_train_N2, W0, W1, cp_inst=cp_N2, alpha=the_alpha, k=the_k, flag=1)
            results_N3, cp_N3 = run_test(data_train_N3, W0, W1, cp_inst=cp_N3, alpha=the_alpha, k=the_k, flag=1)
            results_N4, cp_N4 = run_test(data_train_N4, W0, W1, cp_inst=cp_N4, alpha=the_alpha, k=the_k, flag=1)
            results_final_N1 = results_final_N1 + results_N1
            results_final_N2 = results_final_N2 + results_N2
            results_final_N3 = results_final_N3 + results_N3
            results_final_N4 = results_final_N4 + results_N4
        #print(datasetName)
        results_N1_round = np.around(results_final_N1*100/test_num, 2)
        results_N2_round = np.around(results_final_N2*100/test_num, 2)
        results_N3_round = np.around(results_final_N3*100/test_num, 2)
        results_N4_round = np.around(results_final_N4*100/test_num, 2)

        output[dataset_id*4] = results_N1_round
        output[dataset_id*4+1] = results_N2_round
        output[dataset_id*4+2] = results_N3_round
        output[dataset_id*4+3] = results_N4_round
        dataset_id = dataset_id + 1
        
    return output
  

if __name__ == "__main__":
    
    the_k = 40
    print("EI-kMeans", str(the_k))
    datasetNameSet = ['2d-U-mean/', "2d-1G-mean/", '2d-1G-var/', '2d-1G-cov/', '2d-2G-mean/', '2d-4G-mean/']
    exp2_evaluation(1, the_k, datasetNameSet)