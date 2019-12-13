# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 2019

@author: Anjin Liu (anjin.liu@uts.edu.au)

Experiment data set for paper submitted to TCYB2019,
Title: Concept Drift Detection via Equal Intensity k-means Space Partitioning
"""

import numpy as np
import os

def generate_2d_U_mean(folder='Exp2_SynDriftDetection\\2d-U-mean', r_seed=1, N_train=5000, n_test=500, test_num=250, delta=0.06):
    
    np.random.seed(r_seed)
    data_folder = folder
    os.makedirs(data_folder, exist_ok=True)
    margin = delta
    
    print('Generate dataset at:', data_folder)
    print('  train size N:', N_train, 'test size n:', n_test, 'drift margin:', margin)
    
    data1 = np.random.uniform(size = [N_train, 2])
    data = data1
    np.savetxt(data_folder+'\\data_train.csv', data, delimiter=',', fmt='%1.5f')
    
    data_b1 = np.random.uniform(size = [N_train, 2])
    data_bootstrap = data_b1
    np.savetxt(data_folder+'\\data_bootstrap.csv', data_bootstrap, delimiter=',', fmt='%1.5f')
    
    for t in range(test_num):
        W0_p1 = np.random.uniform(size = [n_test, 2])
        W0 = W0_p1
        W1_p1 = np.random.uniform(size = [n_test, 2])+margin
        W1 = W1_p1
        np.savetxt(data_folder+'\\data_w0_'+str(t)+'.csv', W0, delimiter=',', fmt='%1.5f')
        np.savetxt(data_folder+'\\data_w1_'+str(t)+'.csv', W1, delimiter=',', fmt='%1.5f')


def generate_2d_1G_mean(folder='Exp2_SynDriftDetection\\2d-1G-mean', r_seed=1, N_train=5000, n_test=500, test_num=250, delta=0.3):
    
    np.random.seed(r_seed)
    data_folder = folder
    os.makedirs(data_folder, exist_ok=True)
    margin = delta
    
    print('Generate dataset at:', data_folder)
    print('  train size N:', N_train, 'test size n:', n_test, 'drift margin:', margin)
    
    data1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], N_train)
    data = data1
    np.savetxt(data_folder+'\data_train.csv', data, delimiter=',', fmt='%1.5f')
    
    data_b1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], N_train*100)
    data_bootstrap = data_b1
    np.savetxt(data_folder+'\data_bootstrap.csv', data_bootstrap, delimiter=',', fmt='%1.5f')
    
    for t in range(test_num):
        W0_p1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n_test)
        W0 = W0_p1
        W1_p1 = np.random.multivariate_normal([0 + margin, 0], [[1, 0], [0, 1]], n_test)
        W1 = W1_p1
        np.savetxt(data_folder+'\data_w0_'+str(t)+'.csv', W0, delimiter=',', fmt='%1.5f')
        np.savetxt(data_folder+'\data_w1_'+str(t)+'.csv', W1, delimiter=',', fmt='%1.5f')
        
def generate_2d_1G_var(folder='Exp2_SynDriftDetection\\2d-1G-var', r_seed=1, N_train=5000, n_test=500, test_num=250, delta=0.2):
    
    np.random.seed(r_seed)
    data_folder = folder
    os.makedirs(data_folder, exist_ok=True)
    margin = delta
    
    print('Generate dataset at:', data_folder)
    print('  train size N:', N_train, 'test size n:', n_test, 'drift margin:', margin)
    
    data1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], N_train)
    data = data1
    np.savetxt(data_folder+'\data_train.csv', data, delimiter=',', fmt='%1.5f')
    
    data_b1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], N_train*100)
    data_bootstrap = data_b1
    np.savetxt(data_folder+'\data_bootstrap.csv', data_bootstrap, delimiter=',', fmt='%1.5f')
    
    for t in range(test_num):
        W0_p1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n_test)
        W0 = W0_p1
        W1_p1 = np.random.multivariate_normal([0, 0], [[1 + margin, 0], [0, 1 + margin]], n_test)
        W1 = W1_p1
        np.savetxt(data_folder+'\data_w0_'+str(t)+'.csv', W0, delimiter=',', fmt='%1.5f')
        np.savetxt(data_folder+'\data_w1_'+str(t)+'.csv', W1, delimiter=',', fmt='%1.5f')
        
def generate_2d_1G_cov(folder='Exp2_SynDriftDetection\\2d-1G-cov', r_seed=1, N_train=5000, n_test=500, test_num=250, delta=0.2):
    
    np.random.seed(r_seed)
    data_folder = folder
    os.makedirs(data_folder, exist_ok=True)
    margin = delta
    
    print('Generate dataset at:', data_folder)
    print('  train size N:', N_train, 'test size n:', n_test, 'drift margin:', margin)
    
    data1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], N_train)
    data = data1
    np.savetxt(data_folder+'\data_train.csv', data, delimiter=',', fmt='%1.5f')
    
    data_b1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], N_train*100)
    data_bootstrap = data_b1
    np.savetxt(data_folder+'\data_bootstrap.csv', data_bootstrap, delimiter=',', fmt='%1.5f')
    
    for t in range(test_num):
        W0_p1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n_test)
        W0 = W0_p1
        W1_p1 = np.random.multivariate_normal([0, 0], [[1, 0 + margin], [0 + margin, 1]], n_test)
        W1 = W1_p1
        np.savetxt(data_folder+'\data_w0_'+str(t)+'.csv', W0, delimiter=',', fmt='%1.5f')
        np.savetxt(data_folder+'\data_w1_'+str(t)+'.csv', W1, delimiter=',', fmt='%1.5f')


def generate_2d_2G_mean(folder='Exp2_SynDriftDetection\\2d-2G-mean', r_seed=1, N_train=5000, n_test=500, test_num=250, delta=0.4):
    
    np.random.seed(r_seed)
    data_folder = folder
    os.makedirs(data_folder, exist_ok=True)
    margin = delta
    
    print('Generate dataset at:', data_folder)
    print('  train size N:', N_train, 'test size n:', n_test, 'drift margin:', margin)
    
    data1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], int(N_train/2))
    data2 = np.random.multivariate_normal([5, 0], [[1, 0], [0, 1]], int(N_train/2))
    data = np.vstack([data1, data2])

    data_b1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], int(N_train/2)*100)
    data_b2 = np.random.multivariate_normal([5, 0], [[1, 0], [0, 1]], int(N_train/2)*100)
    data_bootstrap = np.vstack([data_b1, data_b2])
    
    np.random.shuffle(data)
    np.random.shuffle(data_bootstrap)
    np.savetxt(data_folder+'\data_train.csv', data, delimiter=',', fmt='%1.5f')
    np.savetxt(data_folder+'\data_bootstrap.csv', data_bootstrap, delimiter=',', fmt='%1.5f')
    
    for t in range(0,test_num):
        W0_p1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], int(n_test/2))
        W0_p2 = np.random.multivariate_normal([5, 0], [[1, 0], [0, 1]], int(n_test/2))
        W0 = np.vstack([W0_p1, W0_p2])
        W1_p1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], int(n_test/2))
        W1_p2 = np.random.multivariate_normal([5 + margin, 0], [[1, 0], [0, 1]], int(n_test/2))
        W1 = np.vstack([W1_p1, W1_p2])
        np.random.shuffle(W0)
        np.random.shuffle(W1)
        np.savetxt(data_folder+'\data_w0_'+str(t)+'.csv', W0, delimiter=',', fmt='%1.5f')
        np.savetxt(data_folder+'\data_w1_'+str(t)+'.csv', W1, delimiter=',', fmt='%1.5f')

def generate_2d_4G_mean(folder='Exp2_SynDriftDetection\\2d-4G-mean', r_seed=1, N_train=5000, n_test=500, test_num=250, delta=0.8):
    
    np.random.seed(r_seed)
    data_folder = folder
    os.makedirs(data_folder, exist_ok=True)
    margin = delta
    
    print('Generate dataset at:', data_folder)
    print('  train size N:', N_train, 'test size n:', n_test, 'drift margin:', margin)
    
    data1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], int(N_train/4))
    data2 = np.random.multivariate_normal([5, 0], [[1, 0], [0, 1]], int(N_train/4))
    data3 = np.random.multivariate_normal([0, 5], [[1, 0], [0, 1]], int(N_train/4))
    data4 = np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], int(N_train/4))
    data = np.vstack([data1, data2, data3, data4])

    data_b1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], int(N_train/4)*100)
    data_b2 = np.random.multivariate_normal([5, 0], [[1, 0], [0, 1]], int(N_train/4)*100)
    data_b3 = np.random.multivariate_normal([0, 5], [[1, 0], [0, 1]], int(N_train/4)*100)
    data_b4 = np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], int(N_train/4)*100)
    data_bootstrap = np.vstack([data_b1, data_b2, data_b3, data_b4])
    
    np.random.shuffle(data)
    np.random.shuffle(data_bootstrap)
    np.savetxt(data_folder+'\data_train.csv', data, delimiter=',', fmt='%1.5f')
    np.savetxt(data_folder+'\data_bootstrap.csv', data_bootstrap, delimiter=',', fmt='%1.5f')
    
    for t in range(0,test_num):
        W0_p1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], int(n_test/4))
        W0_p2 = np.random.multivariate_normal([5, 0], [[1, 0], [0, 1]], int(n_test/4))
        W0_p3 = np.random.multivariate_normal([0, 5], [[1, 0], [0, 1]], int(n_test/4))
        W0_p4 = np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], int(n_test/4))
        W0 = np.vstack([W0_p1, W0_p2, W0_p3, W0_p4])
        W1_p1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], int(n_test/4))
        W1_p2 = np.random.multivariate_normal([5, 0], [[1, 0], [0, 1]], int(n_test/4))
        W1_p3 = np.random.multivariate_normal([0, 5], [[1, 0], [0, 1]], int(n_test/4))
        W1_p4 = np.random.multivariate_normal([5 - margin, 5], [[1, 0], [0, 1]], int(n_test/4))
        W1 = np.vstack([W1_p1, W1_p2, W1_p3, W1_p4])
        np.random.shuffle(W0)
        np.random.shuffle(W1)
        np.savetxt(data_folder+'\data_w0_'+str(t)+'.csv', W0, delimiter=',', fmt='%1.5f')
        np.savetxt(data_folder+'\data_w1_'+str(t)+'.csv', W1, delimiter=',', fmt='%1.5f')
        
        
def generate_Hd_1G_mean(folder='Exp2_SynDriftDetection\\2d-1G-mean', r_seed=1, N_train=5000, n_test=500, test_num=250, delta=0.3, d=5):
    
    folder = folder.replace('2d-1G-mean', str(d)+'d-1G-mean')
    
    np.random.seed(r_seed)
    data_folder = folder
    os.makedirs(data_folder, exist_ok=True)
    margin = delta
    
    print('Generate dataset at:', data_folder)
    print('  train size N:', N_train, 'test size n:', n_test, 'drift margin:', margin)
    
    mu = np.zeros(d)
    drift_margin = np.zeros(d)
    drift_margin[1] = delta
    mu_drift = mu + drift_margin
    sigma = np.identity(d)
    data1 = np.random.multivariate_normal(mu, sigma, N_train)
    data = data1
    np.savetxt(data_folder+'\data_train.csv', data, delimiter=',', fmt='%1.5f')
    
    data_b1 = np.random.multivariate_normal(mu, sigma, N_train*100)
    data_bootstrap = data_b1
    np.savetxt(data_folder+'\data_bootstrap.csv', data_bootstrap, delimiter=',', fmt='%1.5f')
    
    for t in range(test_num):
        W0_p1 = np.random.multivariate_normal(mu, sigma, n_test)
        W0 = W0_p1
        W1_p1 = np.random.multivariate_normal(mu_drift, sigma, n_test)
        W1 = W1_p1
        np.savetxt(data_folder+'\data_w0_'+str(t)+'.csv', W0, delimiter=',', fmt='%1.5f')
        np.savetxt(data_folder+'\data_w1_'+str(t)+'.csv', W1, delimiter=',', fmt='%1.5f')

    
def generate_Hd_4G_mean(folder='Exp2_SynDriftDetection\\2d-4G-mean', r_seed=1, N_train=5000, n_test=500, test_num=250, delta=0.8, d=5):
    
    folder = folder.replace('2d-4G-mean', str(d)+'d-4G-mean')
    np.random.seed(r_seed)
    data_folder = folder
    os.makedirs(data_folder, exist_ok=True)
    margin = delta
    
    print('Generate dataset at:', data_folder)
    print('  train size N:', N_train, 'test size n:', n_test, 'drift margin:', margin)
    
    mu_1 = np.zeros(d)
    mu_2 = np.zeros(d)
    mu_2[0] = 5
    mu_3 = np.zeros(d)
    mu_3[1] = 5
    mu_4 = np.zeros(d)
    mu_4[0] = 5
    mu_4[1] = 5
    
    drift_margin = np.zeros(d)
    drift_margin[0] = delta
    mu_4_drift = mu_4 - drift_margin
    sigma = np.identity(d)
    
    data1 = np.random.multivariate_normal(mu_1, sigma, int(N_train/4))
    data2 = np.random.multivariate_normal(mu_2, sigma, int(N_train/4))
    data3 = np.random.multivariate_normal(mu_3, sigma, int(N_train/4))
    data4 = np.random.multivariate_normal(mu_4, sigma, int(N_train/4))
    data = np.vstack([data1, data2, data3, data4])


    data_b1 = np.random.multivariate_normal(mu_1, sigma, int(N_train/4)*100)
    data_b2 = np.random.multivariate_normal(mu_2, sigma, int(N_train/4)*100)
    data_b3 = np.random.multivariate_normal(mu_3, sigma, int(N_train/4)*100)
    data_b4 = np.random.multivariate_normal(mu_4, sigma, int(N_train/4)*100)
    data_bootstrap = np.vstack([data_b1, data_b2, data_b3, data_b4])
    data_bootstrap
    # -4， 8 *12-4
    noise_dimension = np.random.random_sample([N_train*100, d-2])
    noise_dimension = noise_dimension*12-4
    data_bootstrap = np.hstack([data_bootstrap, noise_dimension])
    
    np.random.shuffle(data)
    np.random.shuffle(data_bootstrap)
    np.savetxt(data_folder+'\data_train.csv', data, delimiter=',', fmt='%1.5f')
    np.savetxt(data_folder+'\data_bootstrap.csv', data_bootstrap, delimiter=',', fmt='%1.5f')
    
    for t in range(0,test_num):
        W0_p1 = np.random.multivariate_normal(mu_1, sigma, int(n_test/4))
        W0_p2 = np.random.multivariate_normal(mu_2, sigma, int(n_test/4))
        W0_p3 = np.random.multivariate_normal(mu_3, sigma, int(n_test/4))
        W0_p4 = np.random.multivariate_normal(mu_4, sigma, int(n_test/4))
        W0 = np.vstack([W0_p1, W0_p2, W0_p3, W0_p4])
        
        W1_p1 = np.random.multivariate_normal(mu_1, sigma, int(n_test/4))
        W1_p2 = np.random.multivariate_normal(mu_2, sigma, int(n_test/4))
        W1_p3 = np.random.multivariate_normal(mu_3, sigma, int(n_test/4))
        W1_p4 = np.random.multivariate_normal(mu_4_drift, sigma, int(n_test/4))
        W1 = np.vstack([W1_p1, W1_p2, W1_p3, W1_p4])
        
        np.random.shuffle(W0)
        np.random.shuffle(W1)
        np.savetxt(data_folder+'\data_w0_'+str(t)+'.csv', W0, delimiter=',', fmt='%1.5f')
        np.savetxt(data_folder+'\data_w1_'+str(t)+'.csv', W1, delimiter=',', fmt='%1.5f')

if __name__ == "__main__":
    
    # number of samples in the training set
    N_train = 5000
    
    # number of sample in the test windws
    n_test = 200
    test_num = 250
    r_seed = 1
    
    generate_2d_U_mean(r_seed=r_seed, N_train=N_train, n_test=n_test, test_num=test_num)
    generate_2d_1G_mean(r_seed=r_seed, N_train=N_train, n_test=n_test, test_num=test_num)
    generate_2d_1G_var(r_seed=r_seed, N_train=N_train, n_test=n_test, test_num=test_num)
    generate_2d_1G_cov(r_seed=r_seed, N_train=N_train, n_test=n_test, test_num=test_num)
    generate_2d_2G_mean(r_seed=r_seed, N_train=N_train, n_test=n_test, test_num=test_num)
    generate_2d_4G_mean(r_seed=r_seed, N_train=N_train, n_test=n_test, test_num=test_num)
    
    generate_Hd_1G_mean(r_seed=r_seed, N_train=N_train, n_test=n_test, test_num=test_num, d=4,  delta=0.4)
    generate_Hd_1G_mean(r_seed=r_seed, N_train=N_train, n_test=n_test, test_num=test_num, d=6, delta=0.4)
    generate_Hd_1G_mean(r_seed=r_seed, N_train=N_train, n_test=n_test, test_num=test_num, d=8, delta=0.4)
    generate_Hd_1G_mean(r_seed=r_seed, N_train=N_train, n_test=n_test, test_num=test_num, d=10, delta=0.4)
    generate_Hd_1G_mean(r_seed=r_seed, N_train=N_train, n_test=n_test, test_num=test_num, d=20, delta=0.4)
   
    generate_Hd_4G_mean(r_seed=r_seed, N_train=N_train, n_test=n_test, test_num=test_num, d=4, delta=1.5)
    generate_Hd_4G_mean(r_seed=r_seed, N_train=N_train, n_test=n_test, test_num=test_num, d=6, delta=1.5)
    generate_Hd_4G_mean(r_seed=r_seed, N_train=N_train, n_test=n_test, test_num=test_num, d=8, delta=1.5)
    generate_Hd_4G_mean(r_seed=r_seed, N_train=N_train, n_test=n_test, test_num=test_num, d=10, delta=1.5)
    generate_Hd_4G_mean(r_seed=r_seed, N_train=N_train, n_test=n_test, test_num=test_num, d=20, delta=1.5)

    
    
    
    
    
    
    
    
    
    
    
    

