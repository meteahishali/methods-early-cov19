#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:43:24 2020

@author: degerli
"""
import numpy as np

def read_fold_early(path_data, fold):
    class1 = np.load(path_data + 'normal.npy') 
    class2 = np.load(path_data + 'covid_early.npy') 
    class3 = np.load(path_data + 'covid_mild.npy')     
    cov = np.concatenate((class2, class3), axis = 0) 
    
    y_class1 = np.zeros((len(class1), 2))
    y_class2 = np.zeros((len(cov), 2))
    y_class1[:, 0] = 1 #normal
    y_class2[:, 1] = 1 #covid
   
    print('Normal Class: ', class1.shape)
    print('COVID-19 Class: ', cov.shape)

    x_train = [None] 
    x_test = [None]
    y_train = [None] 
    y_test = [None]

    # The below part is to get the k-th fold of the stratified 5-fold cross-validation.
    # Alternatively, you may use sklearn.model_selection.StratifiedKFold which preserves
    # the percentage of samples for each class (approximately). We manually construct the folds
    # for a better approximation.
    testSize1 = 2509 # Control group.
    testSize2 = 160
    testSize3 = 53
    testSize4 = testSize2 + testSize3 # COVID-19 cases.
        
    if fold == 4:
        class1_temp = class1[:]
        class2_temp = class2[:]
        class3_temp = class3[:]
        y_class1_temp = y_class1[:]
        y_class2_temp = y_class2[:]
    
        #Normal - Class1
        x_test = class1[fold * testSize1 : ((fold + 1) * testSize1) - 1, :, :, :]
        y_test = y_class1[fold * testSize1 : ((fold + 1) * testSize1) - 1, :]
        x_train =  np.delete(class1_temp, range(fold * testSize1, ((fold + 1) * testSize1) - 1), axis = 0)
        y_train =  np.delete(y_class1_temp, range(fold * testSize1, ((fold + 1) * testSize1) - 1), axis = 0)
        
        #COVID-19 - Class2    
        x_test = np.concatenate((x_test, class2[fold * testSize2 : ((fold + 1) * testSize2) + 1, :, :, :]), axis = 0)
        x_test = np.concatenate((x_test, class3[fold * testSize3 : ((fold + 1) * testSize3) - 1, :, :, :]), axis = 0)
        y_test = np.concatenate((y_test, y_class2[fold * testSize4 : (fold + 1) * testSize4, :]), axis = 0)    
        x_train = np.concatenate((x_train, np.delete(class2_temp, range(fold * testSize2, ((fold + 1) * testSize2) + 1), axis = 0)), axis = 0)
        x_train = np.concatenate((x_train, np.delete(class3_temp, range(fold * testSize3, ((fold + 1) * testSize3) - 1), axis = 0)), axis = 0)
        y_train =  np.concatenate((y_train, np.delete(y_class2_temp, range(fold * testSize4, (fold + 1) * testSize4), axis = 0)), axis = 0)
        
    else:
        class1_temp = class1[:]
        class2_temp = class2[:]
        class3_temp = class3[:]
        y_class1_temp = y_class1[:]
        y_class2_temp = y_class2[:]
    
        #Normal - Class1
        x_test = class1[fold * testSize1 : (fold + 1) * testSize1, :, :, :]
        y_test = y_class1[fold * testSize1 : (fold + 1) * testSize1, :]
        x_train =  np.delete(class1_temp, range(fold * testSize1, (fold + 1) * testSize1), axis = 0)
        y_train =  np.delete(y_class1_temp, range(fold * testSize1, (fold + 1) * testSize1), axis = 0)
        
        #COVID-19 - Class2    
        x_test = np.concatenate((x_test, class2[fold * testSize2 : (fold + 1) * testSize2, :, :, :]), axis = 0)
        x_test = np.concatenate((x_test, class3[fold * testSize3 : (fold + 1) * testSize3, :, :, :]), axis = 0)
        y_test = np.concatenate((y_test, y_class2[fold * testSize4 : (fold + 1) * testSize4, :]), axis = 0)    
        x_train = np.concatenate((x_train, np.delete(class2_temp, range(fold * testSize2, (fold + 1) * testSize2), axis = 0)), axis = 0)
        x_train = np.concatenate((x_train, np.delete(class3_temp, range(fold * testSize3, (fold + 1) * testSize3), axis = 0)), axis = 0)
        y_train =  np.concatenate((y_train, np.delete(y_class2_temp, range(fold * testSize4, (fold + 1) * testSize4), axis = 0)), axis = 0)
    

    x_train = np.concatenate((x_train, x_train, x_train), axis = 3)
    x_test = np.concatenate((x_test, x_test, x_test), axis = 3)

    return x_train, y_train, x_test, y_test