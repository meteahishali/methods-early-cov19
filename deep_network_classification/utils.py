import tensorflow as tf
from sklearn.metrics import confusion_matrix

from augmentation import *

def performAugmentation(x_train_fold, y_train_fold):
    datagen = augmentation()
    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(np.expand_dims(x_train_fold[:, :, :, 0], axis = -1))
    x_train_fold, y_train_fold = augment(datagen, x_train_fold, y_train_fold)
    print('Train Shape: ', x_train_fold.shape)
    print('Train Label Shape: ', y_train_fold.shape)

    return x_train_fold, y_train_fold

def evaluate(model, x_test_fold, y_test_fold):
    y_predict = np.argmax(model.predict(x_test_fold), axis = 1)
    label = np.argmax(y_test_fold, axis = 1)
        
    CM = confusion_matrix(label, y_predict)
    accuracy = (CM[0,0] + CM[1,1])/sum(sum(CM))
    sensitivity = CM[1,1] / (CM[1,1] + CM[1,0])
    specificity = CM[0,0] / (CM[0,0] + CM[0,1])
    
    return accuracy, sensitivity, specificity, CM

def get_model(modelName):
    input = tf.keras.layers.Input(shape=(224, 224, 3))
    input_shape = (224, 224, 3)
    # Load the model
    if modelName == 'DenseNet121':
        modelTmp = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', 
                pooling='avg', input_tensor = input, input_shape = input_shape)
    elif modelName == 'ResNet50':
        modelTmp = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', 
                pooling='avg', input_tensor = input, input_shape = input_shape)
    elif modelName == 'InceptionV3':
        modelTmp = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', 
                pooling='avg', input_tensor = input, input_shape = input_shape)
    else: exit('Please provide a valid model name.')

    x = modelTmp.output
    predictions = tf.keras.layers.Dense(2, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=input, outputs=predictions)

    return model