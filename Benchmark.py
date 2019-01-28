

# General Porpouse Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import time
import os
import pickle
import datetime
from joblib import Parallel, delayed


# Data loading Libraries
from Libs.Data_loading.dataset_load import on_off_load


# Benchmark functions
from Libs.Benchmark_Libs import  bench, refined_bench, compute_m_v



# To avoid MKL inefficient multythreading
os.environ['MKL_NUM_THREADS'] = '1'

# Simultaneus threads you want to utilise on your machine 
threads = 50

# Plotting settings
sns.set()


# Number of runs 
runs = 50


#%% ON OFF Dataset
# Two class of recordings are used. The first class is composed by files containing
# a single word each, "ON", the second class is equal but the spelled word is "OFF"
# =============================================================================
# number_files_dataset : the number of files to be loaded for each class (On, Off)
# train_test_ratio: ratio between the amount of files used to train
#                                 and test the algorithm, 0.5 will mean that the 
#                                 half of the files wiil be used for training.
# use_all_addr : if False all off events will be dropped, and the total addresses
#                number will correspond to the number of channel of the cochlea
# =============================================================================

number_files_dataset = 60
train_test_ratio = 0.75
use_all_addr = False
number_of_labels = 2

shuffle_seed = 0 # (the seed won't be used if 0)

parameter_folder = "Parameters/On_Off/"
result_folder = "Results/On_Off/"
labels = ("On","Off") 

dataset = Parallel(n_jobs=threads)(delayed(on_off_load)(number_files_dataset, train_test_ratio, shuffle_seed, use_all_addr) for run in range(runs))
[_,_,labels_train, labels_test, filenames_train, labels_train]=np.transpose(dataset)
#%% Load Networks parameter saved from the Playground
file_name = parameter_folder+"GordoNN_Params_1L_8_2019-01-28 15:33:18.295700.pkl"
with open(file_name, 'rb') as f:
    [basis_number, context_lengths, input_channels, taus_T, taus_2D] = pickle.load(f)   
net_parameters = [basis_number, context_lengths, input_channels, taus_T, taus_2D]   
#%% Execute benchmark
start_time = time.time()
bench_results = Parallel(n_jobs=threads)(delayed(bench)(dataset[run],net_parameters,len(labels)) for run in range(runs))   
elapsed_time = time.time()-start_time
print("Learning elapsed time : "+str(elapsed_time))
[prediction_rates,nets]=np.transpose(bench_results)
#%% Take the best and refine the results
best_net = np.argmax(np.sum(prediction_rates,axis=0))
best_prediction_rates =  Parallel(n_jobs=threads)(delayed(refined_bench)(nets[best_net],dataset[run],len(labels)) for run in range(runs))   
#%% Compute mean and variance of the scores of each nework
mean,var = compute_m_v(best_prediction_rates)
#%% Plots
#x = range(3)
#distances = ('Euclidean','Normalised Euclidean','Bhattacharya')            
#fig, ax = plt.subplots()
#plt.bar(x,mean*100,yerr=var*100)
#plt.xticks(x,distances)
#plt.ylim(0,100)
#ax.yaxis.set_major_formatter(PercentFormatter())
#ax.set_title(file_name+" Parameters")
#plt.show()
#%% Save Results
now=datetime.datetime.now()
res_file_name='1L_8_'+str(now)+'.pkl'
with open(result_folder+res_file_name, 'wb') as f:
    pickle.dump([file_name,mean,var,prediction_rates,best_prediction_rates], f)
#%% Save Datasets
now=datetime.datetime.now()
res_file_name='Dataset_1L_8_'+str(now)+'.pkl'
with open(result_folder+res_file_name, 'wb') as f:
    pickle.dump([labels_train, labels_test, filenames_train, labels_train,
                 number_files_dataset,train_test_ratio], f)
#%% Load Results
#res_file_name='2019-01-25 22:12:37.441920.pkl'
#result_folder = "Results/On_Off/"
#with open(result_folder+res_file_name, 'rb') as f:
#       results = pickle.load(f)
#[file_name,mean,var,prediction_rates,best_prediction_rates]= results
