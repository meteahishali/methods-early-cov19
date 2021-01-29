# -*- coding: utf-8 -*-
"""
=========================================================================
Multi-Layer Perceptron classification
Written by Mete Ahishali and Aysen Degerli, Tampere University, Finland.
=========================================================================
"""
import os
import tensorflow as tf
import numpy as np
import scipy.io as sio

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

os.environ["CUDA_VISIBLE_DEVICES"] = '1'  

np.random.seed(7)

def normalizer(x):
    scaler = StandardScaler().fit(x)
    x_normalized = scaler.transform(x)
    
    return x_normalized

#Load the Model#
def call_model(input_shape):
    model = tf.keras.Sequential([      
            tf.keras.layers.Input(shape = (input_shape)),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation = "relu"),
            tf.keras.layers.Dense(2, activation = "softmax")
            ])

    return model

path_net = 'Models/'
if not os.path.exists(path_net): os.makedirs(path_net)
path_data = '../features/'

normalization = True

#Train the model#
accuracy_test = np.zeros((5, ))
specificity = np.zeros((5, ))
sensitivity = np.zeros((5, ))

text_file = open('Results_MLP.txt', "w")
for fold in range(0, 5):
    filepath = path_net + 'model_MLP_fold' + str(fold+1) + '.h5'

    #Read data#
    Data_train = sio.loadmat(path_data + 'features_train_fold_' + str(fold+1) + '.mat')
    Data_test = sio.loadmat(path_data + 'feautures_test_fold_' + str(fold+1) + '.mat')
    label_train = sio.loadmat(path_data + 'y_train_fold_' + str(fold+1) + '.mat')
    label_test = sio.loadmat(path_data + 'y_test_fold_' + str(fold+1) + '.mat')

    x_train_fold = np.array(Data_train['features_train_fold'].astype('float32'))
    y_train_fold = np.array(label_train['y_train_fold'].astype('float32'))
    x_test_fold = np.array(Data_test['features_test_fold'].astype('float32'))
    y_test_fold = np.array(label_test['y_test_fold'].astype('float32'))

    #find pca matrix#
    pca = PCA(n_components = int(x_train_fold.shape[1]/2))
    pca.fit(x_train_fold)

    if normalization:
        x_train_fold = normalizer(x_train_fold) 
        x_test_fold = normalizer(x_test_fold)
    
    model = call_model(x_train_fold.shape[1])
    
    #initialize first layer with pca matrix#
    M_T = [pca.components_.T, np.zeros((int(x_train_fold.shape[1]/2),))]
    model.layers[0].set_weights(M_T)  #Dense_1
    
    model.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(lr = 0.00001), metrics=['accuracy'])
    model.summary()

    history = model.fit(x_train_fold, y_train_fold, epochs = 10, batch_size = 32, verbose = 1, shuffle = True)
    model.save(filepath)
        
    tmp = model.predict(x_test_fold)
    if normalization:
        # MLP becomes oversensitive to covid samples if normalization is applied.
        y_predict = np.zeros([len(x_test_fold), ])
        y_predict[tmp[:, 1] > 0.99997] = 1.0
    else: y_predict = np.argmax(tmp, axis = 1) # Alternatively, normalization can be set false in #Line41.

    label = np.argmax(y_test_fold, axis = 1)
        
    CM = confusion_matrix(label, y_predict)
    total = sum(sum(CM))
    accuracy_test[fold] = (CM[0,0] + CM[1,1])/total
    sensitivity[fold] = CM[1,1] / (CM[1,1] + CM[1,0])
    specificity[fold] = CM[0,0] / (CM[0,0] + CM[0,1])
                
    text_file.write("\nFold: " + str(fold+1) + "\n")
    text_file.write("\nConfusion Matrix: " + str(CM) + "\n")
    
text_file.write("\nAccuracy: " + str(accuracy_test.mean()) + "\n")
text_file.write ("Sensitivity: " + str(sensitivity.mean()) + "\n")
text_file.write ("Specificity: " + str(specificity.mean()) + "\n")
text_file.write("\n\n")

for i in range(0, 5):
    text_file.write("\n\nAccuracy: " + str(accuracy_test[i]) + ", Set: " + str(i+1) + "\n")
    text_file.write ("Sensitivity: " + str(sensitivity[i]) + ", Set: " + str(i+1) + "\n")
    text_file.write ("Specificity: " + str(specificity[i]) + ", Set: " + str(i+1) + "\n")
        
text_file.close()
