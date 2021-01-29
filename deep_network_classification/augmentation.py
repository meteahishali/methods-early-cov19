# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:53:31 2020

@author: degerli
"""
from tqdm import tqdm
import numpy as np
import tensorflow as tf
np.random.seed(7)
tf.random.set_seed(7)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
def augmentation():
    datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=10,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=False,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

    return datagen

def augment(datagen, x_train, y_train):

    print('Data augmentation ...')

    classNumber = y_train.shape[-1] # Number of the classes.
    
    y_labels = np.zeros(y_train.shape[0], dtype = 'float32') # Obtain the labels.
    for i in range(0, len(y_labels)):
        y_labels[i] = np.argmax(y_train[i, :])
    
    NuAugmented = np.zeros(classNumber, dtype = 'float32') # To count how many samples will be augmented for each class.

    samplesNumber = np.zeros(classNumber, dtype = 'float32') # Count how many samples we have for each class.
    for i in range(0, classNumber):
        samplesNumber[i] = np.sum(y_labels == i)
    
    maxSamples = np.max(samplesNumber) # Max number of samples that a class has (data is not balanced).
    # These will be the augmented dataset.
    #filename =  'temp.dat   '
    #x_train_temp = np.memmap(filename, dtype='float32', mode='w+', shape = x_train.shape)
    x_train_temp = x_train
    y_train_temp = y_train

    for i in range(0, classNumber): # Make augmentation for the class i.
        
        x_temp = x_train[y_labels == i, :, :, 0] # Class i samples.
        y_temp = y_train[y_labels == i, :] # Labels for i.

        augmentated_samples = (maxSamples - samplesNumber[i]) # Required augmentation for this class.
        imageGen = datagen.flow(np.expand_dims(x_temp, axis = -1), batch_size=1, seed = 1)
        
        #file = 'x_temp_augmented.data'
        #x_temp_augmented = np.memmap(file, dtype='float32', mode='w+', shape = (int(augmentated_samples), x_temp.shape[1], x_temp.shape[2], 3))
        x_temp_augmented = np.zeros([int(augmentated_samples), x_temp.shape[1], x_temp.shape[2], 3], dtype='float32')
        y_temp_augmented = np.zeros([int(augmentated_samples), classNumber], dtype = 'float32')
        total = 0 # Augmentation counter.
        with tqdm(total=augmentated_samples) as pbar:
            for image in imageGen: # Augmented samples.
                '''
                plt.subplot(1, 2, 1)
                plt.imshow(np.squeeze(x_train[i, :, :, 0]))
                plt.title('Original Image')
                plt.subplot(1, 2, 2)
                plt.imshow(np.squeeze(image))
                plt.title('Augmented Image')
                plt.show()
                '''
                pbar.update()
                # Combine total augmented samples so far with the dataset.
                if total == augmentated_samples:
                    x_train_temp = np.concatenate([x_train_temp, x_temp_augmented], axis = 0)
                    y_train_temp = np.concatenate([y_train_temp, y_temp_augmented], axis = 0)
                    break
                
                x_temp_augmented[total, :, :, 0] = np.squeeze(image)
                x_temp_augmented[total, :, :, 1] = np.squeeze(image)
                x_temp_augmented[total, :, :, 2] = np.squeeze(image)
                y_temp_augmented[total, :] = y_temp[0, :]
                
                total += 1
                NuAugmented[i] += 1 # Keep record how many samples are augmented for the specific classo.            

    for i in range(0, classNumber):
        print('Number of samples before augmentation: ' + str(samplesNumber[i]) + ' for class ' + str(i))

    print('\n\n')
    for i in range(0, classNumber):
        print('Number of samples after augmentation: ' + str(np.sum(y_train_temp[:, i] == 1)) + ' for class ' + str(i))
        print(str(NuAugmented[i]) + ' number of samples are augmented.\n')
        
    return x_train_temp, y_train_temp