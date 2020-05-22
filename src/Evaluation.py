import numpy as np
from Functions import *

path_root = './activity/'
trained_sample = '50000'
testing_num = 10000
assignment_num = 10000
power = 0.1

#------------------------------------------------------------------------------ 
# Load neurons activity
testing_result_monitor = np.load(path_root + 'Activity_testing_A1_' + str(testing_num) + trained_sample + '.npy')
assignment_result_monitor = np.load(path_root + 'Activity_assignment_A1_' + str(assignment_num) + trained_sample + '.npy')

testing_input_numbers = np.load(path_root + 'Labels_testing' + str(testing_num) + trained_sample + '.npy')
assignment_input_numbers = np.load(path_root + 'Labels_assignment' + str(assignment_num) + trained_sample + '.npy')
    
#------------------------------------------------------------------------------ 
# Assignments
print 'get assignments(1 class) for A1'
assignments = get_new_assignments(assignment_result_monitor, assignment_input_numbers)
print 'get assignments(10 class) for A1'
assignments_for_10 = get_new_assignments_for_10(assignment_result_monitor, assignment_input_numbers, power)

#------------------------------------------------------------------------------ 
# Accuracy

print '\ncalculate accuracy for A1'
results_proportion = np.zeros((10, assignment_num))
test_results = np.zeros((10, assignment_num))
for j in range(assignment_num):
    results_proportion[:,j] = get_recognized_number_proportion(assignments, assignment_result_monitor[j,:])
    test_results[:,j] = np.argsort(results_proportion[:,j])[::-1]
difference = test_results[0,:] - assignment_input_numbers[:]
correct = len(np.where(difference == 0)[0])
incorrect = len(np.where(difference != 0)[0])
sum_accurracy = correct/float(assignment_num) * 100
print 'Assignment for 1 class Accuracy(validating): ', sum_accurracy, ' num incorrect: ', incorrect
    
results_proportion = np.zeros((10, testing_num))
test_results = np.zeros((10, testing_num))
for j in range(testing_num):
    results_proportion[:,j] = get_recognized_number_proportion(assignments, testing_result_monitor[j,:])
    test_results[:,j] = np.argsort(results_proportion[:,j])[::-1]
difference = test_results[0,:] - testing_input_numbers[:]
correct = len(np.where(difference == 0)[0])
incorrect = len(np.where(difference != 0)[0])
sum_accurracy = correct/float(testing_num) * 100
print 'Assignment for 1 class Accuracy(testing): ', sum_accurracy, ' num incorrect: ', incorrect
    
results_proportion = np.zeros((10, assignment_num))
test_results = np.zeros((10, assignment_num))
for j in range(assignment_num):
    results_proportion[:,j] = get_recognized_number_proportion_for_10(assignments_for_10, assignment_result_monitor[j,:])
    test_results[:,j] = np.argsort(results_proportion[:,j])[::-1]
difference = test_results[0,:] - assignment_input_numbers[:]
correct = len(np.where(difference == 0)[0])
incorrect = len(np.where(difference != 0)[0])
sum_accurracy = correct/float(assignment_num) * 100
print 'Assignment for 10 classes(power:', power, ') Accuracy(validating): ', sum_accurracy, ' num incorrect: ', incorrect
    
results_proportion = np.zeros((10, testing_num))
test_results = np.zeros((10, testing_num))
for j in range(testing_num):
    results_proportion[:,j] = get_recognized_number_proportion_for_10(assignments_for_10, testing_result_monitor[j,:])
    test_results[:,j] = np.argsort(results_proportion[:,j])[::-1]
difference = test_results[0,:] - testing_input_numbers[:]
correct = len(np.where(difference == 0)[0])
incorrect = len(np.where(difference != 0)[0])
sum_accurracy = correct/float(testing_num) * 100
print 'Assignment for 10 classes(power:', power, ') Accuracy(testing): ', sum_accurracy, ' num incorrect: ', incorrect
