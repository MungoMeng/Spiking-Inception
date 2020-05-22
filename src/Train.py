import numpy as np
import os.path
import scipy 
import brian2 as b
from brian2 import *
import time
from math import *

from Functions import *

#set parameters
np.random.seed(0)
prefs.codegen.target = 'auto'
prefs.codegen.cpp.extra_compile_args_gcc = ['-march=native']

#---------------------------------------------------Build network----------------------------------------------------------
Learning = True

print 'The settings of network are as follow:'
print '---------------------------------------------'

#the amount of input
n_input = 784

#connection parameters
kernel_type_num = 3
kernel_size_each_type = [28, 24, 16]
stride_each_type = [1, 4, 6]
feature_map_size_each_type = [int(((sqrt(n_input) - kernel_size_each_type[i]) / stride_each_type[i] + 1)**2) for i in range(kernel_type_num)]
print 'Num of kernel type:', kernel_type_num
print 'Kernel size and Stride of each kernel type:', kernel_size_each_type, stride_each_type
print 'Feature map size of each kernel type:', feature_map_size_each_type

feature_map_num = 448
kernel_num_each_type = [4, 2, 1]
kernel_num = np.sum(kernel_num_each_type)
print 'Feature map num of each module:', feature_map_num
print 'Kernel num of each kernel type:', kernel_num_each_type
print 'Num of kernel:', kernel_num

#the amount of neurons
neuron_num_each_kernel = []
feature_map_size_each_kernel = []
kernel_size_each_kernel = []
stride_each_kernel = []
for kernel_type in range(kernel_type_num):
    for kernel in range(kernel_num_each_type[kernel_type]):
        neuron_num_each_kernel.append(feature_map_num * feature_map_size_each_type[kernel_type])
        feature_map_size_each_kernel.append(feature_map_size_each_type[kernel_type])
        kernel_size_each_kernel.append(kernel_size_each_type[kernel_type])
        stride_each_kernel.append(stride_each_type[kernel_type])

neuron_num = np.sum(neuron_num_each_kernel)
print 'Neurons num of each kernel:', neuron_num_each_kernel
print 'Num of Neurons:', neuron_num
print '---------------------------------------------'
    
#neuron parameters
v_rest_e = -65. * b.mV
v_reset_e = -65. * b.mV
v_thresh_e = -52. * b.mV
refrac_e = 5. * b.ms

#synapses parameters
Delay = 10*b.ms
tc_pre_ee = 20*b.ms
tc_post_1_ee = 20*b.ms
tc_post_2_ee = 40*b.ms
nu_ee_pre =  0.0001
nu_ee_post = 0.01
wmax_ee = 1.0
wmin_ee = 1e-7
ihn = 24
norm = 78.4
    
if Learning == False:
    scr_e = 'v = v_reset_e'
else:
    tc_theta = 1e7 * b.ms
    theta_plus_e = 0.05 * b.mV
    scr_e = 'v = v_reset_e; theta += theta_plus_e'
offset = 20.0*b.mV
thresh_e = 'v>(theta - offset + ' + str(v_thresh_e/b.mV) + '*mV' + ')'

#equation of excitatory neurons
neuron_eqs_e = '''
            dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt
            I_synE = ge * nS * (        -v)                             : amp
            I_synI = gi * nS * (-100.*mV-v)                             : amp
            dge/dt = -ge/(1.0*ms)                                       : 1
            dgi/dt = -gi/(2.0*ms)                                       : 1
            '''
if Learning == False:
    neuron_eqs_e += '\n  theta      :volt'
else:
    neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'

#equation of STDP
eqs_stdp_ee = '''
                    w                                      : 1
                    post2before                            : 1
                    dpre/dt    = -pre/(tc_pre_ee)          : 1 (event-driven)
                    dpost1/dt  = -post1/(tc_post_1_ee)     : 1 (event-driven)
                    dpost2/dt  = -post2/(tc_post_2_ee)     : 1 (event-driven)
                '''
eqs_stdp_pre_ee = 'ge+=w; pre = 1.; w = (w>0)*clip(w - nu_ee_pre * post1 , wmin_ee, wmax_ee)'
eqs_stdp_post_ee = 'post2before = post2; w = (w>0)*clip(w + nu_ee_post * pre * post2before, wmin_ee, wmax_ee); post1 = 1.; post2 = 1.'
    
#create empty dict
neuron_groups = {}
connections = {}
spike_counters = {}
net = {}
    
#create neuron group
neuron_groups['X1'] = b.PoissonGroup(n_input, 0*b.hertz)
neuron_groups['A1'] = b.NeuronGroup(neuron_num, neuron_eqs_e, method='euler', threshold=thresh_e, refractory=refrac_e, reset= scr_e)
neuron_groups['A1'].v = v_rest_e - 40. * b.mV
neuron_groups['A1'].theta = np.ones((neuron_num)) * 20.0*b.mV
    
#create connections AA
start = time.time()
weightMatrix = np.zeros((neuron_num, neuron_num))
mark = 0
for kernel in range(kernel_num):
    feature_map_size = feature_map_size_each_kernel[kernel]
    for src in range(mark, mark+neuron_num_each_kernel[kernel]):
        S = src - mark
        src_z = int(S/feature_map_size)
        src_y = int((S - src_z*feature_map_size) / sqrt(feature_map_size))
        src_x = int(S - src_z*feature_map_size - src_y*sqrt(feature_map_size))
        for tar in range(mark, mark+neuron_num_each_kernel[kernel]):
            T = tar - mark
            tar_z = int(T / feature_map_size)
            tar_y = int((T - tar_z*feature_map_size) / sqrt(feature_map_size))
            tar_x = int(T - tar_z*feature_map_size - tar_y*sqrt(feature_map_size))
            if src_x == tar_x and src_y == tar_y and src_z != tar_z:
                weightMatrix[src,tar] = ihn
    mark += neuron_num_each_kernel[kernel]
weightMatrix = weightMatrix.reshape((neuron_num*neuron_num))
connections['A1A1'] = b.Synapses(neuron_groups['A1'], neuron_groups['A1'], 'w:1',on_pre='gi+=w')
connections['A1A1'].connect()
connections['A1A1'].w = weightMatrix
end = time.time()
print 'time needed to create connection A1A1:', end - start

#create connections XA
start = time.time()
weightMatrix = np.zeros((n_input, neuron_num))
if Learning:
    mark = 0
    for kernel in range(kernel_num):
        feature_map_size = feature_map_size_each_kernel[kernel]
        kernel_size = kernel_size_each_kernel[kernel]
        stride = stride_each_kernel[kernel]
        for src in range(n_input):
            src_z = int(src / n_input)
            src_y = int((src - src_z*n_input) / sqrt(n_input))
            src_x = int(src - src_z*n_input - src_y*sqrt(n_input))
            for tar in range(mark, mark+neuron_num_each_kernel[kernel]):
                T = tar - mark
                tar_z = int(T / feature_map_size)
                tar_y = int((T - tar_z*feature_map_size) / sqrt(feature_map_size))
                tar_x = int(T - tar_z*feature_map_size - tar_y*sqrt(feature_map_size))
                if src_x >= tar_x*stride and src_x < tar_x*stride+kernel_size and src_y >= tar_y*stride and src_y < tar_y*stride+kernel_size:
                    weightMatrix[src,tar] = 0.3*rand()+wmin_ee
        mark += neuron_num_each_kernel[kernel]
weightMatrix = weightMatrix.reshape((n_input*neuron_num))

if Learning:
    connections['X1A1'] = b.Synapses(neuron_groups['X1'], neuron_groups['A1'], eqs_stdp_ee, on_pre=eqs_stdp_pre_ee, on_post=eqs_stdp_post_ee)
else:
    connections['X1A1'] = b.Synapses(neuron_groups['X1'], neuron_groups['A1'], 'w : 1', on_pre='ge+=w')
connections['X1A1'].connect()
connections['X1A1'].w = weightMatrix
connections['X1A1'].delay = 'rand()*'+ str(Delay/b.ms) +'*ms'
end = time.time()
print 'time needed to create connection X1A1:', end - start

#create monitors
spike_counters['A1'] = b.SpikeMonitor(neuron_groups['A1'], record=False)

#create networks
net['M1'] = Network(neuron_groups['A1'],neuron_groups['X1'],connections['X1A1'],connections['A1A1'],spike_counters['A1'])
#-----------------------------------------------------------------------------------------------------------------------

# load MNIST
start = time.time()
training = get_labeled_data('./training')
end = time.time()
print 'time needed to load training set:', end - start

#specify the location
save_path = './weights/'
load_path = './weights/'

#the time-window of simulation
single_example_time =   0.35 * b.second
resting_time = 0.15 * b.second

#the the interval of process data and show information
progress_interval = 10
validate_interval = 5000   #no less than 2000
save_interval = 500

#number of samples for training
n_train = 60000
train_begin = 0    #specify which iteration you want the training to begin from 

#load trained weight to continue
if train_begin:
    connections['X1A1'].w = np.load(load_path + 'X1A1' + '_' + str(train_begin) + '.npy')
    neuron_groups['A1'].theta = np.load(load_path + 'theta_A1' + '_' + str(train_begin) + '.npy') *b.volt

#the intensity of rate coding
intensity_step = 0.125
start_intensity = 0.25

#the threshold of retrain
retrain_gate = np.sum([5*feature_map_size_each_kernel[kernel] for kernel in range(kernel_num)])

# run the simulation and set inputs
previous_spike_count = {}
current_spike_count = {}
assignments = {}
result_monitor = {}
results_proportion = {}
accuracy = {}

previous_spike_count['A1'] = np.zeros(neuron_num)
current_spike_count['A1'] = np.zeros(neuron_num)
assignments['A1'] = np.zeros(neuron_num)
result_monitor['A1'] = np.zeros((validate_interval,neuron_num))
results_proportion['A1'] = np.zeros((10, validate_interval))
accuracy['A1'] = []
input_numbers = np.zeros(validate_interval)

neuron_groups['X1'].rates = 0*b.hertz
net['M1'].run(0*b.second)

start = time.time()
j = train_begin
input_intensity = start_intensity
while j < n_train:

    Rates = training['x'][j%60000,:,:].reshape((n_input)) * input_intensity

    neuron_groups['X1'].rates = Rates*b.hertz
    connections['X1A1'] = normalize_weights(connections['X1A1'],norm)

    net['M1'].run(single_example_time)
    
    current_spike_count['A1'] = np.asarray(spike_counters['A1'].count[:])- previous_spike_count['A1']
    previous_spike_count['A1'] = np.copy(spike_counters['A1'].count[:])
    
    #if current_spike_count is not enough, increase the input_intensity and simulat this example again
    spike_num = np.sum(current_spike_count['A1'])
    #print spike_num

    if spike_num < retrain_gate:
        input_intensity += intensity_step
        neuron_groups['X1'].rates = 0*b.hertz
        net['M1'].run(resting_time)
    else:
        result_monitor['A1'][j%validate_interval,:] = current_spike_count['A1']
        input_numbers[j%validate_interval] = training['y'][j%60000][0]

        neuron_groups['X1'].rates = 0*b.hertz
        net['M1'].run(resting_time)
        input_intensity = start_intensity

        j += 1
        if j % progress_interval == 0:
            print 'Progress: ', j, '/', n_train, '(', time.time() - start, 'seconds)'
            start = time.time()
        if j % validate_interval == 0:
            assignments['A1'] = get_new_assignments(result_monitor['A1'][:], input_numbers[:])
            test_results = np.zeros((10, validate_interval))
            for k in range(validate_interval):
                results_proportion['A1'][:,k] = get_recognized_number_proportion(assignments['A1'], result_monitor['A1'][k,:])
                test_results[:,k] = np.argsort(results_proportion['A1'][:,k])[::-1]
            difference = test_results[0,:] - input_numbers[:]
            correct = len(np.where(difference == 0)[0])
            accuracy['A1'].append(correct/float(validate_interval) * 100)
            print 'Validate accuracy: ', accuracy['A1'][-1], '(last)', np.max(accuracy['A1']), '(best)'
            
        if j % save_interval == 0:
            np.save(save_path + 'X1A1' + '_' + str(j), connections['X1A1'].w)
            np.save(save_path + 'theta_A1' + '_' + str(j), neuron_groups['A1'].theta)




