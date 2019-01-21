import numpy as np
from Libs.Solid_HOTS.Solid_HOTS_Network import Solid_HOTS_Net




def bench(dataset,net_parameters,number_of_labels):
    net_seed = 0 #Network creation is complitely randomic
    exploring = False # The network won'tgenerate any message
    dataset_learning, labels_learning, dataset_testing, labels_testing = dataset
    [basis_number, context_lengths, input_channels, taus_T, taus_2D] = net_parameters   

    # Generate the network
    Net = Solid_HOTS_Net(basis_number, context_lengths, input_channels, taus_T,
                          taus_2D, exploring, net_seed)
    
    Net.learn(dataset_learning)
    Net.histogram_classification_train(labels_learning,number_of_labels)
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
    return mean,var