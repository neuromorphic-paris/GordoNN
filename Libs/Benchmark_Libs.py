import numpy as np
from Libs.Solid_HOTS.Solid_HOTS_Network import Solid_HOTS_Net

# Data loading Libraries
from Libs.Data_loading.dataset_load import on_off_load


def bench(dataset_parameters, network_parameters, classifier_parameters):
    
    [number_files_dataset, label_file,
     train_test_ratio, shuffle_seed, use_all_addr] = dataset_parameters
    
    [[features_number, context_lengths, input_channels, taus_T, taus_2D, 
                 nthreads, exploring],
     [learning_rate, epochs, l1_norm_coeff, cross_correlation_th_array, 
      batch_size, spacing]] = network_parameters
    
    [[last, number_of_labels, mlp_epochs,  mlp_learning_rate],[threshold]] = classifier_parameters
    
    [dataset_train, dataset_test, classes_train, classes_test, filenames_train,
     filenames_test, wordpos_train, wordpos_test] = on_off_load(number_files_dataset,
                                                                label_file,
                                                                train_test_ratio,
                                                                shuffle_seed,
                                                                use_all_addr)

    
    # Generate the network
    Net = Solid_HOTS_Net(features_number, context_lengths, input_channels,
                         taus_T, taus_2D, nthreads, exploring)
    
    Net.cross_correlation_th_array = cross_correlation_th_array 
    Net.batch_size = batch_size
    Net.spacing = spacing

    Net.learn(dataset_train,dataset_test,learning_rate, epochs, l1_norm_coeff)
    
    
    Net.mlp_epochs = mlp_epochs
    
    [labels, labels_test] = Net.mlp_single_word_classification_train(classes_train,
                                                                     classes_test,
                                                                     wordpos_train,
                                                                     wordpos_test,
                                                                     number_of_labels,
                                                                     mlp_learning_rate,
                                                                     last)
    
    [prediction_rates, predicted_labels,
     net_activity] = Net.mlp_single_word_classification_test(classes_test,
                                                           number_of_labels,
                                                           threshold, last)    
    single_run_results = prediction_rates
    filenames=[filenames_train, filenames_test]
    
    return single_run_results, filenames

def refined_bench(Net,dataset,number_of_labels):
    dataset_learning, dataset_testing, labels_learning, labels_testing, filenames_train, filenames_test = dataset
    prediction_rate, distances, predicted_labels = Net.histogram_classification_test(labels_testing,number_of_labels,dataset_testing)
        
    single_run_results = prediction_rate
        
    return single_run_results

def compute_m_v(bench_results):
    runs = len(bench_results)
    mean=np.zeros(3) 
    var=np.zeros(3)     
    for run in range(runs):
        mean+=bench_results[run]
    mean = mean/runs
    for run in range(runs):
        var += (bench_results[run]-mean)**2
    var = var/runs
    return mean,var