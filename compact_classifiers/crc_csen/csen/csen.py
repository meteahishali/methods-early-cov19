import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import numpy as np
import argparse

## Functions
from utils_csen import *
from networks import *

np.random.seed(10) # numpy is good about making repeatable output
tf.random.set_seed(10) # on the other hand, this is basically useless (see issue 9171)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('--test', default=False, help="evaluate the model")
args = vars(ap.parse_args())
weights = args['test']

dataPath = '../CSENdata/'
weightsDir = 'weights/'
outResults = 'csen_Results.txt'

imageSizeM = 50
imageSizeN = 25
class_No = 2

##### Network and training parameters
input_shape = (imageSizeM, imageSizeN, 1)
param = {'batch_size':32, 'epochs':15, 'filepath':'foo'}

metrics = {'accuracy':np.zeros([5, 2]), 'sensitivity':np.zeros([5, 2]), 
            'specificity':np.zeros([5, 2]), 'cofMatrices':[]}

for set in range(1, 6):
    filepath1 = weightsDir + 'weights_CNN_' + str(set) + '.h5'
    filepath2 = weightsDir + 'weights_CNN2_' + str(set) + '.h5'

    x_train, y_train, x_test, y_test = loadData(dataPath, set)

    # Build the CNN model
    CNNmodel = get_CNN(input_shape)
    CNNmodel2 = get_CNN2(input_shape)

    if weights is False:
        param['filepath'] = filepath1
        train(CNNmodel, param, x_train, y_train)
        param['filepath'] = filepath2
        train(CNNmodel2, param, x_train, y_train)

    cm1,metrics['accuracy'][set-1,0],metrics['sensitivity'][set-1,0],metrics['specificity'][set-1,0]=evaluate(
        filepath1, CNNmodel, x_test, y_test)
    cm2,metrics['accuracy'][set-1,1],metrics['sensitivity'][set-1,1],metrics['specificity'][set-1,1]=evaluate(
        filepath2, CNNmodel2, x_test, y_test)

    # Confusion matrix.
    metrics['cofMatrices'].append(cm1)
    metrics['cofMatrices'].append(cm2)

writeResults(outResults, metrics)