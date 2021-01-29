import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import numpy as np
import argparse

## Functions
from utils_reconnet import *
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
outResults = 'reconnet_Results.txt'

imageSizeM = 50
imageSizeN = 25
class_No = 2

##### Network and training parameters
input_shape = (imageSizeM, imageSizeN, 1)
param = {'batch_size':32, 'epochs':15, 'filepath':'foo'}

metrics = {'accuracy':np.zeros([5, ]), 'sensitivity':np.zeros([5, ]), 
            'specificity':np.zeros([5, ]), 'cofMatrices':[]}

for set in range(1, 6):
    filepath = weightsDir + 'weights_Reconnet_' + str(set) + '.h5'

    x_train, y_train, x_test, y_test = loadData(dataPath, set)

    # Build the CNN model
    Reconnet = get_Reconnet(input_shape)

    if weights is False:
        param['filepath'] = filepath
        train(Reconnet, param, x_train, y_train)

    cm,metrics['accuracy'][set-1],metrics['sensitivity'][set-1],metrics['specificity'][set-1]=evaluate(
        filepath, Reconnet, x_test, y_test)

    # Confusion matrix.
    metrics['cofMatrices'].append(cm)

writeResults(outResults, metrics)