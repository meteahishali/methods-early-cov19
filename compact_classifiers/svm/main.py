"""
=========================================================================
SVM classification
Written by Mete Ahishali and Aysen Degerli, Tampere University, Finland.
=========================================================================
"""

import numpy as np
from sklearn.metrics import confusion_matrix

# Functions
from utils import * 


path = '../features/'
#Save the results
file = open('SVM_results.txt', "w")

accuracy = np.zeros((5, 1))
sensitivity = np.zeros((5, 1))
specificity = np.zeros((5, 1))

# You can set this False since we already recorded the best SVM parameters.
parameterSearch = False

for i in range(0, 5):
    print('\nFold ' + str(i + 1) + ' ...')
    x_train, y_train, x_test, y_test = loadData(path, i + 1)

    if parameterSearch:
        ### If hyper-parameter search
        print('Parameter search is selected.')
        best_parameters, best_model = SVM_train_search(x_train, y_train)
        score = best_model.score(x_test, y_test)
        y_predict = best_model.predict(x_test)
        file.write('\nBest paramters:' + str(best_parameters))
    else:
        svm_model = SVM_train(x_train, y_train)
        y_predict = svm_model.predict(x_test)

    cm = confusion_matrix(y_test, y_predict)
    print('\nConfusion Matrix: \n', cm)
    total = sum(sum(cm))
    accuracy[i] = (cm[0,0] + cm[1,1])/total
    sensitivity[i] = cm[1,1] / (cm[1,1] + cm[1,0])
    specificity[i] = cm[0,0] / (cm[0,0] + cm[0,1])
    print('Accuracy : ', accuracy[i])
    print('Sensitivity : ', sensitivity) # TP / (TP + FN)
    print('Specificity : ', specificity) # TN / (TN + FP)

    file.write("\n\nFold " + str(i + 1) + " :\n")
    file.write("\nAccuracy of: " + str((cm[0,0] + cm[1,1])/total) + "\n")
    file.write ("Sensitivity of: " + str(cm[1,1] / (cm[1,1] + cm[1,0])) + "\n")
    file.write ("Specificity of: " + str(cm[0,0] / (cm[0,0] + cm[0,1])) + "\n")

file.write("\n\nAveraged metrics:\n")
file.write("Accuracy of: " + str(accuracy.mean()) + "\n")
file.write ("Sensitivity of: " + str(sensitivity.mean()) + "\n")
file.write ("Specificity of: " + str(specificity.mean()) + "\n")

file.close()