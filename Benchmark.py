

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
from Libs.Benchmark_Libs import  generate_nets, bench, refined_bench, compute_m_v



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

number_files_dataset = 20
train_test_ratio = 0.75
use_all_addr = False
number_of_labels = 2

shuffle_seed = 0 # (the seed won't be used if 0)

parameter_folder = "Parameters/On_Off/"
result_folder = "Results/On_Off/"
labels = ("On","Off") 

dataset = Parallel(n_jobs=threads)(delayed(on_off_load)(number_files_dataset, train_test_ratio, shuffle_seed, use_all_addr) for run in range(runs))

#%% Load Networks parameter saved from the Playground
file_name = parameter_folder+"GordoNN_Params_2019-01-22 13:24:17.367409.pkl"
with open(file_name, 'rb') as f:
    [basis_number, context_lengths, input_channels, taus_T, taus_2D] = pickle.load(f)   
net_parameters = [basis_number, context_lengths, input_channels, taus_T, taus_2D]   
#%% Execute benchmark
start_time = time.time()
nets = Parallel(jobs=threads)(delayed(generate_nets)(net_parameters) for run in range(runs))
bench_results = Parallel(n_jobs=threads)(delayed(bench)(nets[run], dataset[run],len(labels)) for run in range(runs))   
elapsed_time = time.time()-start_time
print("Learning elapsed time : "+str(elapsed_time))

#%% Take the best and refine the results
best_net = np.argmax(np.sum(bench_results,axis=1))
refined_bench_results =  Parallel(n_jobs=threads)(delayed(refined_bench)(nets[best_net],dataset[run],net_parameters,len(labels)) for run in range(runs))   
#%% Compute mean and variance of the scores of each nework
mean,var = compute_m_v(refined_bench_results)
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
res_file_name=str(now)+'.pkl'
with open(result_folder+res_file_name, 'wb') as f:
    pickle.dump([file_name,mean,var,bench_results,refined_bench_results,best_net], f)
#%% Load Results
#res_file_name='2019-01-22 16:16:32.804131.pkl'
#result_folder = "Results/On_Off/"
#with open(result_folder+res_file_name, 'rb') as f:
#       results = pickle.load(f)
#[file_name,mean,var,bench_results,refined_bench_results,best_net]= results