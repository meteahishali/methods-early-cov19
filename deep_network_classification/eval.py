# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:53:31 2020

Training and evaluation of several deep network models on Early-QaTa-COV19.

Author: Aysen Degerli and Mete Ahishali,
Tampere University, Tampere, Finland.
"""

import os
import tensorflow as tf
import numpy as np
import argparse

from read_data import *
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True, help="network model: DenseNet121, ResNet50, InceptionV3")
ap.add_argument('--test', default=False, help="evaluate the model")
args = vars(ap.parse_args())
weights = args['test']
modelName = args['model']

np.random.seed(7)

path_net = 'Models/'
path_data = '../data/'
DataAugmentation = True

#Train the model#
metrics = {'accuracy':np.zeros([5, 1]), 'sensitivity':np.zeros([5, 1]), 
            'specificity':np.zeros([5, 1])}

text_file = open('Results_' + modelName + '.txt', 'w')
for fold in range(0, 5):

    #Read data#
    x_train_fold, y_train_fold, x_test_fold, y_test_fold = read_fold_early(path_data, fold)  
    
    if DataAugmentation:
        x_train_fold, y_train_fold = performAugmentation(x_train_fold, y_train_fold)
    
    # Normalize data.
    x_train_fold = x_train_fold / 255
    x_test_fold = x_test_fold / 255
    
    tf.random.set_seed(4)
    #Load the Model#
    filepath = path_net + 'model_' + modelName + '_fold' + str(fold+1) + '.h5'
    if weights: model = tf.keras.models.load_model(filepath)
    else: model = get_model(modelName)

    model.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(lr = 0.00001), metrics=['accuracy'])
    model.summary()
    
    #Train the network
    if weights is False:
        history = model.fit(x_train_fold, y_train_fold, epochs = 10, batch_size = 32, verbose = 1, shuffle = True)
        model.save(filepath)
    
    accuracy, sensitivity, specificity, CM = evaluate(model, x_test_fold, y_test_fold)
    metrics['accuracy'][fold] = accuracy
    metrics['sensitivity'][fold] = sensitivity
    metrics['specificity'][fold] = specificity
                
    text_file.write("\nFold: " + str(fold+1) + "\n")
    text_file.write("\nConfusion Matrix: \n" + str(CM) + "\n")
    
    del model, x_train_fold, y_train_fold, x_test_fold, y_test_fold
    
text_file.write("\nAccuracy of: " + str(metrics['accuracy'].mean()) + "\n")
text_file.write ("Sensitivity of: " + str(metrics['sensitivity'].mean()) + "\n")
text_file.write ("Specificity of: " + str(metrics['specificity'].mean()) + "\n")
text_file.write("\n\n")

for i in range(0, 5):
    text_file.write("\n\nAccuracy of: " + str(metrics['accuracy'][i]) + ", Set: " + str(i+1) + "\n")
    text_file.write ("Sensitivity of: " + str(metrics['sensitivity'][i]) + ", Set: " + str(i+1) + "\n")
    text_file.write ("Specificity of: " + str(metrics['specificity'][i]) + ", Set: " + str(i+1) + "\n")
        
text_file.close()
