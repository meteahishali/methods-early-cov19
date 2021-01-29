import numpy as np
import scipy.io

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from numba import jit, prange

######################################## FUNCTIONS
def loadData(dataPath, set):

    Data = scipy.io.loadmat(dataPath + 'data_dic_0.5_' + str(set) + '.mat')

    x_train = Data["x_train"].astype('float64')
    x_test = Data["x_test"].astype('float64')
    x_dic = Data["x_dic"].astype('float64')
    y_train = Data["y_train"].astype('float64')
    y_test = Data["y_test"].astype('float64')
    y_dic = Data["y_dic"].astype('float64')

    x_train = np.concatenate([x_train, x_dic], axis = 0)
    y_train = np.concatenate([y_train, y_dic], axis = 0)

    # Standardize data (0 mean, 1 stdev)
    x_train = normalizer(x_train)
    x_test = normalizer(x_test)

    print(x_train.shape)
    print(x_test.shape)
   
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    y_train = np.expand_dims(y_train, axis=3)
    y_test = np.expand_dims(y_test, axis=3)

    class_No = len(np.unique(y_train))
   
    dic_label = scipy.io.loadmat(dataPath + 'dic_label.mat')["ans"]

    y_train = labelCreator(dic_label, y_train, class_No)
    y_test = labelCreator(dic_label, y_test, class_No)

    #one-hot encode target column
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test

def train(model, param, x_train, y_train):
    checkpoint = ModelCheckpoint(param['filepath'], monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.fit(x_train, y_train, epochs = param['epochs'], batch_size = param['batch_size'], shuffle = True, callbacks=callbacks_list)

def evaluate(filepath, model, x_test, y_test):
    model.load_weights(filepath)
    
    label = np.argmax(y_test, axis = 1)
    tmp = model.predict(x_test)
    y_predict = np.zeros([len(x_test), ])
    y_predict[tmp[:, 1] >= 1] = 1.0 # ReconNet is oversensitive to covid samples.

    cm = confusion_matrix(label, y_predict)
    total = sum(sum(cm))
    accuracy = (cm[0,0] + cm[1,1])/total
    sensitivity = cm[1,1] / (cm[1,1] + cm[1,0]) # TP / (TP + FN)
    specificity = cm[0,0] / (cm[0,0] + cm[0,1]) # TN / (TN + FP)

    return cm, accuracy, sensitivity, specificity

def writeResults(outResults, metrics):

    cofMatrices = metrics['cofMatrices']
    accuracy = metrics['accuracy']
    sensitivity = metrics['sensitivity']
    specificity = metrics['specificity']
    text_file = open(outResults, "w")

    for i in range(0, 5):
        text_file.write('\nConfusion Matrices for Set ' + str(i + 1))
        text_file.write('\nReconNet: \n')
        text_file.write(str(cofMatrices[i]))
        text_file.write('\n')

    text_file.write("\nReconNet Accuracy: " + str(accuracy.mean()) + "\n")
    text_file.write ("ReconNet Sensitivity: " + str(sensitivity.mean()) + "\n")
    text_file.write ("ReconNet Specificity :" + str(specificity.mean()) + "\n")

    text_file.write("\n\n")
    for i in range(0, 5):
        text_file.write("\nReconNet Accuracy: " + str(accuracy[i]) + ", Set: " + str(i+1) + "\n")
        text_file.write ("ReconNet Sensitivity: " + str(sensitivity[i]) + ", Set: " + str(i+1) + "\n")
        text_file.write ("ReconNet Specificity: " + str(specificity[i]) + ", Set: " + str(i+1) + "\n")
    text_file.close()

def normalizer(x):
    m =  x.shape[1]
    n =  x.shape[2]
    x = np.reshape(x, [len(x), m * n])
    scaler = StandardScaler().fit(x)
    x_normalized = scaler.transform(x)
    x_normalized = np.reshape(x_normalized, [len(x_normalized), m, n])
    
    return x_normalized

@jit(forceobj=True, parallel=True)
def labelCreator(dic_label, y, class_No):
    
    label = np.zeros(len(y))

    for i in range(len(y)):
        arr_true = y[i, :, :, 0]
        
        energies_true = np.zeros([class_No, ])
        for c in range(0, class_No):
            mask = dic_label == (c + 1)
            result_true = arr_true * mask
            energies_true[c] = result_true.sum()
                
        label[i]=np.argmax(energies_true)

    return label