# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:53:31 2020

Training DenseNet-121 on Early-QaTa-COV19, and feature extraction for
the compact classifier approaches.

Author: Aysen Degerli,
Tampere University, Tampere, Finland.
"""

import os
import tensorflow as tf
import numpy as np
import scipy.io as sio
import argparse

from read_data import *
from augmentation import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('--weights', default=False, help="evaluate the model")
args = vars(ap.parse_args())
weights = args['weights']

np.random.seed(7)

path_net = 'Models/'
path_data = '../data/'
outdir = 'features/'
if not os.path.exists(path_net): os.makedirs(path_net)
if not os.path.exists(outdir): os.makedirs(outdir)

DataAugmentation = True

score = np.zeros((5, 1))

for fold in range(0, 5):

    #Read data#
    x_train_fold, y_train_fold, x_test_fold, y_test_fold = read_fold_early(path_data, fold)  
    
    if DataAugmentation:
        datagen = augmentation()
        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(np.expand_dims(x_train_fold[:, :, :, 0], axis = -1))
        x_train_fold, y_train_fold = augment(datagen, x_train_fold, y_train_fold)
        print('Train Shape: ', x_train_fold.shape)
        print('Train Label Shape: ', y_train_fold.shape)
        
    # Normalize data.
    x_train_fold = x_train_fold / 255
    x_test_fold = x_test_fold / 255
    x_train_mean = np.mean(x_train_fold, axis=0)
    x_train_fold -= x_train_mean
    x_test_fold -= x_train_mean

    filepath = path_net + 'model_DenseNet121_fold' + str(fold+1) + '.h5'

    # Load the model
    if weights is False:
        densenet121 = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', 
                pooling='avg', input_shape = (224, 224, 3), classes=2)

        x = densenet121.output
        predictions = tf.keras.layers.Dense(2, activation="softmax", name="predictions")(x)
        model = tf.keras.Model(inputs=densenet121.input, outputs=predictions)
    else: model = tf.keras.models.load_model(filepath)
    
    model.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(lr = 0.00001), metrics=['accuracy'])
    model.summary()
    
    #Train the network
    if weights is False:
        history = model.fit(x_train_fold, y_train_fold, epochs = 10, batch_size = 32, verbose = 1, shuffle = True)
        model.save(filepath)
    
    # Extract Features
    model_features = tf.keras.Model(inputs = model.input,
                           outputs = model.get_layer(name = 'avg_pool', index = None).output)
    features_train = model_features.predict(x_train_fold, verbose = 1)
    features_test = model_features.predict(x_test_fold, verbose = 1) 
     
    # Save the data and extracted features
    sio.savemat(outdir + 'y_train_fold_' + str(fold+1) + '.mat', mdict={'y_train_fold': y_train_fold})
    sio.savemat(outdir + 'y_test_fold_' + str(fold+1) + '.mat', mdict={'y_test_fold': y_test_fold})

    sio.savemat(outdir + 'features_train_fold_' + str(fold+1) + '.mat', mdict={'features_train_fold': features_train})
    sio.savemat(outdir + 'feautures_test_fold_' + str(fold+1) + '.mat', mdict={'features_test_fold': features_test})
    
    if DataAugmentation: del datagen
    
    del x_train_fold, y_train_fold, x_test_fold, y_test_fold, model, model_features
