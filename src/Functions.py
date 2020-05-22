import numpy as np
import time
import os.path
import scipy 
import cPickle as pickle
import brian2 as b
from struct import unpack
from brian2 import *
from brian2tools import *

MNIST_data_path = '/home/public/data/MNIST/'

#------------------------------------------------------------------------------ 
# functions
#------------------------------------------------------------------------------  
def get_labeled_data(picklename, bTrain = True):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename))
    else:
        # Open the images with gzip in read binary mode
        if bTrain:
            images = open(MNIST_data_path + 'train-images-idx3-ubyte','rb')
            labels = open(MNIST_data_path + 'train-labels-idx1-ubyte','rb')
        else:
            images = open(MNIST_data_path + 't10k-images-idx3-ubyte','rb')
            labels = open(MNIST_data_path + 't10k-labels-idx1-ubyte','rb')
        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = unpack('>I', images.read(4))[0]
        rows = unpack('>I', images.read(4))[0]
        cols = unpack('>I', images.read(4))[0]
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack('>I', labels.read(4))[0]
    
        if number_of_images != N:
            raise Exception('number of labels did not match the number of images')
        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
        for i in xrange(N):
            if i % 1000 == 0:
                print("i: %i" % i)
            x[i] = [[unpack('>B', images.read(1))[0] for unused_col in xrange(cols)]  for unused_row in xrange(rows) ]
            y[i] = unpack('>B', labels.read(1))[0]
            
        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open("%s.pickle" % picklename, "wb"))
    return data

def normalize_weights(connection,norm):
    n_input = connection.source.N
    n_e = connection.target.N
    temp_conn = np.copy(connection.w)
    temp_conn = temp_conn.reshape((n_input,n_e))
    colSums = np.sum(temp_conn, axis = 0)
    colFactors = norm/colSums
    for j in xrange(n_e):
        temp_conn[:,j] *= colFactors[j]
    connection.w = temp_conn.reshape((n_input*n_e))
    return connection

def get_new_assignments(result_monitor, input_numbers):
    #print result_monitor.shape
    n_e = result_monitor.shape[1]
    assignments = np.ones(n_e) * -1 # initialize them as not assigned
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e    
    rate = [0] * n_e    
    for j in xrange(10):
        num_inputs = len(np.where(input_nums == j)[0])
        if num_inputs > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_inputs
        for i in xrange(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j 
    return assignments

def get_new_assignments_for_10(result_monitor, input_numbers, power):
    #print result_monitor.shape
    #print input_numbers.shape
    n_e = result_monitor.shape[1]
    assignments = np.zeros((n_e,10)) # initialize them as not assigned
    rate = np.zeros((10,n_e))
    count = np.zeros((10))
    for n in range(input_numbers.shape[0]):
        rate[input_numbers[n],:] += result_monitor[n,:]
        count[input_numbers[n]] += 1
    for n in range(10):
        rate[n,:] = rate[n,:] / count[n]   
    for n in range(n_e):
        rate_power = np.power(rate[:,n], power)
        if np.sum(rate_power) > 0:
            assignments[n,:] = [rate_power[i]/np.sum(rate_power) for i in range(10)]
    return assignments

def get_recognized_number_proportion(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in range(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    summed_proportion = summed_rates/ np.sum(summed_rates)
    return summed_proportion

def get_recognized_number_proportion_for_10(assignments_for_10, spike_rates):
    summed_rates = [0] * 10
    for i in range(10):
        summed_rates[i] = np.sum(spike_rates * assignments_for_10[:,i]) / len(spike_rates)
    summed_proportion = summed_rates/ np.sum(summed_rates)
    return summed_proportion

    
