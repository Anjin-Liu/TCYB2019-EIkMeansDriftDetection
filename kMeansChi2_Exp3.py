# -*- coding: utf-8 -*-
"""
Created on Fri May 31 17:23:04 2019

@author: Anjin Liu (anjin.liu@uts.edu.au)

Experiment data set for paper submitted to TCYB2019,
Title: Concept Drift Detection via Equal Intensity k-means Space Partitioning
"""

import numpy as np
import kMeansChi2_lib

def kMeansChi2_test(data_train, W0, W1, kMeansChi2_inst=None, alpha=0.05, k=16, output_path_partition=None):
    
    if kMeansChi2_inst is None:
        kMeansChi2_inst = kMeansChi2_lib.kMeansChi2(k)
        kMeansChi2_inst.buildkMeans(data_train, output_path=output_path_partition)
        
        results = np.zeros(2)
        results[0] = kMeansChi2_inst.drift_detection(W0, alpha)
        results[1] = kMeansChi2_inst.drift_detection(W1, alpha)

        return results, kMeansChi2_inst
    else:
        results = np.zeros(2)
        results[0] = kMeansChi2_inst.drift_detection(W0, alpha)
        results[1] = kMeansChi2_inst.drift_detection(W1, alpha)
        
        return results, kMeansChi2_inst

def kMeansChi2_exp3(r_seed, the_k, dataFolderPath_Main, datasetNameSet, alpha=0.05, test_num=250):
    np.random.seed(r_seed)
    the_alpha = alpha
    output = np.zeros([len(datasetNameSet), 2])
    dataset_id = 0
    for datasetName in datasetNameSet:
        dataFolderPath = dataFolderPath_Main + datasetName
        data_train = np.loadtxt(dataFolderPath+'data_train.csv', dtype=float, delimiter=',')
        data_train_N1 = data_train
        results_final_N1 = np.zeros(2)
        kChi2_N1 = None
        for t in range(test_num):
            W0 = np.loadtxt(dataFolderPath+'data_W0_'+str(t)+'.csv', dtype=float, delimiter=',')
            W1 = np.loadtxt(dataFolderPath+'data_W1_'+str(t)+'.csv', dtype=float, delimiter=',')
            results_N1, kChi2_N1 = kMeansChi2_test(data_train_N1, W0, W1, kMeansChi2_inst=kChi2_N1, alpha=the_alpha, k=the_k)
            results_final_N1 = results_final_N1 + results_N1
        results_N1_round = np.around(results_final_N1*100/test_num, 2)
        output[dataset_id] = results_N1_round
        dataset_id = dataset_id + 1
    return output
  
def Exp3_run_all(r_seed):
    
    the_k = 40
    print("kMeansChi2 Higgs", str(the_k))
    dataFolderPath_Main = 'Datasets/Exp3_ReaDriftDetection/1_Higgs/Data/'
    datasetNameSet = ['back_higg/', "higg_back/"]
    higg_result = kMeansChi2_exp3(r_seed, the_k, dataFolderPath_Main, datasetNameSet)
    
    the_k = 40
    print("kMeansChi2 MiniBooNe", str(the_k))
    dataFolderPath_Main = 'Datasets/Exp3_ReaDriftDetection/2_MiniBooNeParticle/Data/'
    datasetNameSet = ['singal_backgr/', "backgr_singal/"]
    mini_result = kMeansChi2_exp3(r_seed, the_k, dataFolderPath_Main, datasetNameSet)
    
    the_k = 40
    print("kMeansChi2 Arabic_Digit", str(the_k))
    dataFolderPath_Main = 'Datasets/Exp3_ReaDriftDetection/3_Arabic_Digit/Data/'
    datasetNameSet = [ "mix1_mix2/", "mix2_mix1/"]
    arab_result = kMeansChi2_exp3(r_seed, the_k, dataFolderPath_Main, datasetNameSet)
    
    the_k = 60
    print("kMeansChi2 Localization", str(the_k))
    dataFolderPath_Main = 'Datasets/Exp3_ReaDriftDetection/4_Localization/Data/'
    datasetNameSet = [ "mix1/"]
    loca_result = kMeansChi2_exp3(r_seed, the_k, dataFolderPath_Main, datasetNameSet)
    
    the_k = 40
    print("kMeansChi2 Insects", str(the_k))
    dataFolderPath_Main = 'Datasets/Exp3_ReaDriftDetection/5_Insects/Data/'
    datasetNameSet = [ "mix1/"]
    inse_result = kMeansChi2_exp3(r_seed, the_k, dataFolderPath_Main, datasetNameSet)
    
    
    t1_result_ds_list = np.zeros(8)
    t2_result_ds_list = np.zeros(8)
    t1_result_ds_list[:2] = higg_result[:,0]
    t2_result_ds_list[:2] = 100-higg_result[:,1]
    t1_result_ds_list[2:4] = mini_result[:,0]
    t2_result_ds_list[2:4] = 100-mini_result[:,1]
    t1_result_ds_list[4:6] = arab_result[:,0]
    t2_result_ds_list[4:6] = 100-arab_result[:,1]
    t1_result_ds_list[6] = loca_result[0][0]
    t2_result_ds_list[6] = 100-loca_result[0][1]
    t1_result_ds_list[7] = inse_result[0][0]
    t2_result_ds_list[7] = 100-inse_result[0][1]
    
    return t1_result_ds_list, t2_result_ds_list

if __name__ == "__main__":
    
    print("kMeansChi2 Exp3")
    r_seed = 0
    t1_result_ds_list, t2_result_ds_list = Exp3_run_all(r_seed)
    
    
    
    
    