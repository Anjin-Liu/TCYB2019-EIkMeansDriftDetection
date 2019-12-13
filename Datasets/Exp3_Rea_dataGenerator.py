# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:40:14 2019

@author: Anjin Liu (anjin.liu@uts.edu.au)

Experiment data set for paper submitted to TCYB2019,
Title: Concept Drift Detection via Equal Intensity k-means Space Partitioning
"""
import scipy.io as sio
import numpy as np
import os

def generate_Higgs(r_seed, file_back, file_higg, datafolder_path, N_train, N_test, num_test):
    print('Generating Higgs-Back and Back-Higgs')
    np.random.seed(r_seed)
    
    data_back = sio.loadmat(file_back)['data0']
    data_higg = sio.loadmat(file_higg)['data1']
    
    
    ################
    # data shuffle #
    ################
    
    np_back = np.asarray(data_back)
    np_higg = np.asarray(data_higg)

    np.random.shuffle(np_back)
    np.random.shuffle(np_higg)
    
    total_samples = N_train + N_test*num_test
    
    np_back = np_back[:total_samples,:-1]
    np_higg = np_higg[:total_samples,:-1]
    
    np_back_data_train = np_back[:N_train]
    np_higg_data_train = np_higg[:N_train]
    
    output_data_folder_bh = datafolder_path + "back_higg"
    os.makedirs(output_data_folder_bh, exist_ok=True)
    np.savetxt(output_data_folder_bh+'/data_train.csv', np_back_data_train, delimiter=',', fmt='%s')
    
    output_data_folder_hb = datafolder_path + "higg_back"
    os.makedirs(output_data_folder_hb, exist_ok=True)
    np.savetxt(output_data_folder_hb+'/data_train.csv', np_higg_data_train, delimiter=',', fmt='%s')
    
    for t in range(N_train, total_samples, N_test):
        
        b = np_back[t:t+N_test]
        h = np_higg[t:t+N_test]
        
        np.savetxt(output_data_folder_bh+'/data_w0_'+str(int((t-N_train)/N_test))+'.csv', b, delimiter=',', fmt='%s')
        np.savetxt(output_data_folder_bh+'/data_w1_'+str(int((t-N_train)/N_test))+'.csv', h, delimiter=',', fmt='%s')
        np.savetxt(output_data_folder_hb+'/data_w0_'+str(int((t-N_train)/N_test))+'.csv', h, delimiter=',', fmt='%s')
        np.savetxt(output_data_folder_hb+'/data_w1_'+str(int((t-N_train)/N_test))+'.csv', b, delimiter=',', fmt='%s')

def generate_MiniBooNe(r_seed, datafolder_path, datafile_name, N_train, N_test, num_test):
    print('Generating Sign-Back and Back-Sign')
    np.random.seed(r_seed)
    
    singal_event_num = 36499
    backgr_event_num = 93565

    file = open(datafile_name, "r")
    
    singal_event_data = []
    backgr_event_data = []
    
    line_counter = 0
    for line in file:
        items = line.split()
        if 0 < line_counter and line_counter <= singal_event_num:
            singal_event_data.append(','.join(items))
        if line_counter > singal_event_num:
            backgr_event_data.append(','.join(items))
        line_counter += 1
    file.close()

    ################
    # data shuffle #
    ################
    singal_event_data = np.array(singal_event_data)
    backgr_event_data = np.array(backgr_event_data)
    
    np.random.shuffle(singal_event_data)
    np.random.shuffle(backgr_event_data)
    
    singal_event_train_idx = np.random.permutation(singal_event_num)[:N_train]
    backgr_event_train_idx = np.random.permutation(backgr_event_num)[:N_train]
    
    singal_event_data_train = singal_event_data[singal_event_train_idx]
    backgr_event_data_train = backgr_event_data[backgr_event_train_idx]
    
    output_data_folder_sb = datafolder_path + "singal_backgr"
    os.makedirs(output_data_folder_sb, exist_ok=True)
    np.savetxt(output_data_folder_sb+'/data_train.csv', singal_event_data_train, fmt='%s')
    
    output_data_folder_bs = datafolder_path + "backgr_singal"
    os.makedirs(output_data_folder_bs, exist_ok=True)
    np.savetxt(output_data_folder_bs+'/data_train.csv', backgr_event_data_train, fmt='%s')
    
    for t in range(num_test):
        sb_W0_idx = np.random.permutation(singal_event_num)[:N_test]
        sb_W1_idx = np.random.permutation(backgr_event_num)[:N_test]
        
        bs_W0_idx = np.random.permutation(backgr_event_num)[:N_test]
        bs_W1_idx = np.random.permutation(singal_event_num)[:N_test]

        W0_sb = singal_event_data[sb_W0_idx]
        W1_sb = backgr_event_data[sb_W1_idx]
        W0_bs = backgr_event_data[bs_W0_idx]
        W1_bs = singal_event_data[bs_W1_idx]
        
        np.savetxt(output_data_folder_sb+'/data_w0_'+str(t)+'.csv', W0_sb, fmt='%s')
        np.savetxt(output_data_folder_sb+'/data_w1_'+str(t)+'.csv', W1_sb, fmt='%s')
        np.savetxt(output_data_folder_bs+'/data_w0_'+str(t)+'.csv', W0_bs, fmt='%s')
        np.savetxt(output_data_folder_bs+'/data_w1_'+str(t)+'.csv', W1_bs, fmt='%s')

def generate_ArabicDigit(r_seed, datafolder_path, datafile_name, N_train, N_test, num_test):
    print('Generating Arabic A-B and B-A')
    np.random.seed(r_seed)
    
    mixture_1 = ['m0','m1','m2','m3','m4','f5','f6','f7','f8','f9']
    mixture_2 = ['m0','m1','m2','m3','m4','f5','f6','f7','f8','m9']
    
    file = open(datafile_name, "r")    
    
    data = {}
    for i in range(len(mixture_1)):
        data['m'+str(i)] = []
    for i in range(len(mixture_1)):
        data['f'+str(i)] = []
    
    
    line_counter = 0
    for line in file:
        items = line.split(',')
        if 0 < line_counter:
            if items[-2]=='male':
                data['m'+items[-1].replace('\n', '')].append(','.join(items[:-2]))
            else:
                data['f'+items[-1].replace('\n', '')].append(','.join(items[:-2]))
        line_counter += 1
    file.close()
    
    ################
    # data shuffle #
    ################
    
    mixture_1_data = []
    mixture_2_data = []
    for item in mixture_1:
        mixture_1_data += data[item]
    for item in mixture_2:
        mixture_2_data += data[item]   
    
    mixture_1_data = np.array(mixture_1_data)
    mixture_2_data = np.array(mixture_2_data)
    mixture_1_num = len(mixture_1_data)
    mixture_2_num = len(mixture_2_data)
    
    idx_array_1 = np.arange(mixture_1_num)
    idx_array_2 = np.arange(mixture_2_num)
    
    np.random.shuffle(idx_array_1)
    np.random.shuffle(idx_array_2)
    mixture_1_train_idx = idx_array_1[:N_train]
    mixture_2_train_idx = idx_array_2[:N_train]
    idx_array_1 = idx_array_1[N_train:]
    idx_array_2 = idx_array_2[N_train:]
    
    mixture_1_data_train = mixture_1_data[mixture_1_train_idx]
    mixture_2_data_train = mixture_2_data[mixture_2_train_idx]
    
    output_data_folder_mf = datafolder_path + "mix1_mix2"
    os.makedirs(output_data_folder_mf, exist_ok=True)
    np.savetxt(output_data_folder_mf+'/data_train.csv', mixture_1_data_train, fmt='%s')
    
    output_data_folder_fm = datafolder_path + "mix2_mix1"
    os.makedirs(output_data_folder_fm, exist_ok=True)
    np.savetxt(output_data_folder_fm+'/data_train.csv', mixture_2_data_train, fmt='%s')
    
    for t in range(num_test):
        
        np.random.shuffle(idx_array_1)
        np.random.shuffle(idx_array_2)
        m1m2_W0_idx = idx_array_1[:N_test]
        m1m2_W1_idx = idx_array_2[:N_test]
        
        np.random.shuffle(idx_array_1)
        np.random.shuffle(idx_array_2)
        m2m1_W0_idx = idx_array_2[:N_test]
        m2m1_W1_idx = idx_array_1[:N_test]

        W0_m1m2 = mixture_1_data[m1m2_W0_idx]
        W1_m1m2 = mixture_2_data[m1m2_W1_idx]
        W0_m2m1 = mixture_2_data[m2m1_W0_idx]
        W1_m2m1 = mixture_1_data[m2m1_W1_idx]
        
        np.savetxt(output_data_folder_mf+'/data_w0_'+str(t)+'.csv', W0_m1m2, fmt='%s')
        np.savetxt(output_data_folder_mf+'/data_w1_'+str(t)+'.csv', W1_m1m2, fmt='%s')
        np.savetxt(output_data_folder_fm+'/data_w0_'+str(t)+'.csv', W0_m2m1, fmt='%s')
        np.savetxt(output_data_folder_fm+'/data_w1_'+str(t)+'.csv', W1_m2m1, fmt='%s')
        
        
def generate_Localization(r_seed, datafolder_path, datafile_name, N_train, N_test, num_test):
    print('Generating Localization')
    np.random.seed(r_seed)
    
    person = ['A','B','C','D','E']
    
    activity = ['lying', 'walking', 'sitting']
    sensor = ['010-000-024-033', '010-000-030-096', '020-000-033-111', '020-000-032-221']    
    data = {}  
    
    file = open(datafile_name, "r")

    for line in file:
        
        line = line.replace('\n','')
        items = line.split(',')
        
        p = items[0][0]
        if items[-1] in activity:
            if p+' '+items[-1]+' '+items[1] in data.keys():
                data[p+' '+items[-1]+' '+items[1]].append(','.join(items[4:-1]))
            else:
                data[p+' '+items[-1]+' '+items[1]] = [','.join(items[4:-1])]
    file.close()
        
    ################
    # data shuffle #
    ################
    
    # Drift Config
    person_per_train = {'A':0,'B':50,'C':50,'D':50,'E':100}
    person_per_test_W0 = {'A':0,'B':10,'C':10,'D':10,'E':20}
    person_per_test_W1 = {'A':20,'B':20,'C':10,'D':0,'E':0}
    
    data_train = {}
    data_test = {}
    for p in person:
        for a in activity:
            for s in sensor:
                k = ' '.join([p, a, s])
                pas_data = np.array(data[k])
                np.random.shuffle(pas_data)
                if ' '.join([p]) in data_train.keys():
                    data_train[' '.join([p])] = np.append(data_train[' '.join([p])], 
                              pas_data[:person_per_train[p]])
                else:
                    data_train[' '.join([p])] = pas_data[:person_per_train[p]]
                data_test[k] = pas_data[person_per_train[p]:]
       
    mix1_train = np.array([])         
    for p in person:
        mix1_train = np.append(mix1_train, data_train[p])
        
    np.random.shuffle(mix1_train)
        
    output_data_folder = datafolder_path + 'mix1'
    os.makedirs(output_data_folder, exist_ok=True)
    np.savetxt(output_data_folder+'/data_train.csv', mix1_train, fmt='%s')
    
    for t in range(num_test):
        data_test_shuffel = {}
        for p in person:
            for a in activity:
                for s in sensor:
                    k = ' '.join([p, a, s])
                    pas_data = np.array(data_test[k])
                    if ' '.join([p]) in data_test_shuffel.keys():
                        data_test_shuffel[' '.join([p])] = np.append(data_test_shuffel[' '.join([p])], 
                                  pas_data[:person_per_test_W0[p]])
                    else:
                        data_test_shuffel[' '.join([p])] = pas_data[:person_per_test_W0[p]]
        
        mix1_test_w0 = np.array([]) 
        for p in person:
            np.random.shuffle(data_test_shuffel[p])
            mix1_test_w0 = np.append(mix1_test_w0, data_test_shuffel[p])
        np.random.shuffle(mix1_test_w0)
        output_data_folder = datafolder_path + 'mix1'
        np.savetxt(output_data_folder+'/data_w0_'+str(t)+'.csv', mix1_test_w0, fmt='%s')
            
        data_test_shuffel = {}
        for p in person:
            for a in activity:
                for s in sensor:
                    k = ' '.join([p, a, s])
                    pas_data = np.array(data_test[k])
                    if ' '.join([p]) in data_test_shuffel.keys():
                        data_test_shuffel[' '.join([p])] = np.append(data_test_shuffel[' '.join([p])], 
                                  pas_data[:person_per_test_W1[p]])
                    else:
                        data_test_shuffel[' '.join([p])] = pas_data[:person_per_test_W1[p]]
                
        mix1_test_w1 = np.array([]) 
        for p in person:
            np.random.shuffle(data_test_shuffel[p])
            mix1_test_w1 = np.append(mix1_test_w1, data_test_shuffel[p])
        np.random.shuffle(mix1_test_w1)
        output_data_folder = datafolder_path + 'mix1'
        np.savetxt(output_data_folder+'/data_w1_'+str(t)+'.csv', mix1_test_w1, fmt='%s')        
        
        
def generate_Insects(r_seed, datafolder_path, datafile_name, N_train, N_test, num_test):
    print('Generating Insect')
    np.random.seed(r_seed)
    
    file = open(datafile_name, "r")    
    insect_type = ['flies', 'aedes', 'tarsalis', 'quinx', 'fruit']
    
    mix1_train_config = {'flies':400, 'aedes':400, 'tarsalis':400, 'quinx':400, 'fruit':400}
    mix1_test_w0_config = {'flies':100, 'aedes':100, 'tarsalis':100, 'quinx':100, 'fruit':100}
    mix1_test_w1_config = {'flies':70, 'aedes':70, 'tarsalis':100, 'quinx':100, 'fruit':160}
    data = {}    
    
    for line in file:
        line = line.replace('\n','')
        items = line.split(',')
        if items[-1] in data:
            data[items[-1]].append(','.join(items[:-1]))
        else:
            data[items[-1]] = [','.join(items[:-1])]
    file.close()
        
    ################
    # data shuffle #
    ################
    
    mix1_train_data = np.array([])
    mix1_test_data = {}
    
    for insect_t in insect_type:
        np.random.shuffle(data[insect_t])
        mix1_train_data = np.append(mix1_train_data, data[insect_t][:mix1_train_config[insect_t]])
        mix1_test_data[insect_t] = data[insect_t][mix1_train_config[insect_t]:]
    
    output_data_folder = datafolder_path + "mix1"
    os.makedirs(output_data_folder, exist_ok=True)
    np.savetxt(output_data_folder+'/data_train.csv', mix1_train_data, fmt='%s')
    
    for t in range(num_test):
        
        w0 = np.array([])
        for insect_t in insect_type:
            np.random.shuffle(mix1_test_data[insect_t])
            w0 = np.append(w0, mix1_test_data[insect_t][:mix1_test_w0_config[insect_t]])
        np.random.shuffle(w0)
        
        w1 = np.array([])
        for insect_t in insect_type:
            np.random.shuffle(mix1_test_data[insect_t])
            w1 = np.append(w1, mix1_test_data[insect_t][:mix1_test_w1_config[insect_t]])
            
        np.random.shuffle(w1)
        np.savetxt(output_data_folder+'/data_w0_'+str(t)+'.csv', w0, fmt='%s')
        np.savetxt(output_data_folder+'/data_w1_'+str(t)+'.csv', w1, fmt='%s')
        
if __name__ == "__main__":
    
    num_test = 250
    r_seed = 1
    
    generate_Higgs(r_seed, "OriginalDataFiles/1_Higgs/data0", "OriginalDataFiles/1_Higgs/data1", "Exp3_ReaDriftDetection/1_Higgs/Data/", 2000, 1000, num_test)
    generate_MiniBooNe(r_seed, "Exp3_ReaDriftDetection/2_MiniBooNeParticle/Data/", "OriginalDataFiles/2_MiniBooNe/MiniBooNE_PID.txt", 2000, 500, num_test)
    generate_ArabicDigit(r_seed, "Exp3_ReaDriftDetection/3_Arabic_Digit/Data/", "OriginalDataFiles/3_Arabic_Digit/ArabicDigit_Shuffled_With_Sex.csv", 2000, 500, num_test)
    generate_Localization(r_seed, "Exp3_ReaDriftDetection/4_Localization/Data/", "OriginalDataFiles/4_Localization/ConfLongDemo_JSI.txt", 3000, 600, num_test)
    generate_Insects(r_seed, "Exp3_ReaDriftDetection/5_Insects/Data/", "OriginalDataFiles/5_Insects/Insects.data", 2000, 500, num_test)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
