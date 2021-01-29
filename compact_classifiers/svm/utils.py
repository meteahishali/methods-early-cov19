import scipy.io as sio
import numpy as np

from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def normalizer(x):
  scaler = StandardScaler().fit(x)
  x_normalized = scaler.transform(x)
    
  return x_normalized

def loadData(path, fold):
  Data_train = sio.loadmat(path + 'features_train_fold_' + str(fold) + '.mat')
  Data_test = sio.loadmat(path + 'feautures_test_fold_' + str(fold) + '.mat')
  label_train = sio.loadmat(path + 'y_train_fold_' + str(fold) + '.mat')
  label_test = sio.loadmat(path + 'y_test_fold_' + str(fold) + '.mat')

  x_train = Data_train['features_train_fold'].astype('float32')
  y_train = label_train['y_train_fold'].astype('float32')
  x_test = Data_test['features_test_fold'].astype('float32')
  y_test = label_test['y_test_fold'].astype('float32')

  
  x_train, y_train = shuffle(x_train, y_train, random_state = 1)

  pca = PCA(n_components = 512, random_state = 1)
  pca.fit(x_train)
  x_train = pca.transform(x_train)
  x_test = pca.transform(x_test)

  #x_train = normalizer(x_train)
  #x_test = normalizer(x_test)
  y_train = np.argmax(y_train, axis = 1)
  y_test = np.argmax(y_test, axis = 1)

  return x_train, y_train, x_test, y_test

def SVM_train(X_train, y_train):
  
  svm_model = SVC(kernel = 'rbf', gamma = 0.001, C = 10, random_state = 1)
  
  print('SVM Train')
  svm_model.fit(X_train, y_train)
  print('SVM Train Finished')
  
  return svm_model

def SVM_train_search(X_train, y_train):
                     
  params_grid = [{'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
                    {'kernel': ['poly'], 'degree': [2, 3, 4], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]             

  svm_model = GridSearchCV(SVC(), params_grid, n_jobs = 8, cv = 3)
  
  print('SVM Train')
  svm_model.fit(X_train, y_train)
  print('SVM Train Finished')
  
  return svm_model.best_params_, svm_model.best_estimator_
