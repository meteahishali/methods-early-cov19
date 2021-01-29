import tensorflow as tf
###### Get CNN model
def get_CNN(input_shape):
    
    input = tf.keras.layers.Input(shape = input_shape, name='input')
    x_0 = tf.keras.layers.Conv2D(48, 3, padding = 'same', activation='relu')(input)
    x_1 = tf.keras.layers.Conv2D(24, 3, padding = 'same', activation='relu')(x_0)
    y = tf.keras.layers.Conv2D(1, 3, padding = 'same', activation='relu')(x_1)

    y = tf.keras.layers.AveragePooling2D(pool_size=(25, 25), strides=None, padding='valid', data_format=None)(y)
    y = tf.keras.layers.Flatten()(y)
    y = tf.keras.layers.Softmax()(y)

    CNNmodel = tf.keras.models.Model(input, y, name='segmentation_network')
    CNNmodel.summary()

    adam = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    CNNmodel.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics=['mae', 'acc'])

    return CNNmodel

def get_CNN2(input_shape):
   
    input = tf.keras.layers.Input(shape = input_shape, name='input')
    x_0 = tf.keras.layers.Conv2D(48, 3, padding = 'same', activation = 'relu')(input)
    x_0 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_0) # Comment for Compact
    x_1 = tf.keras.layers.Conv2DTranspose(24, 3, strides=(2, 2), padding='same', activation = 'relu')(x_0) # Comment for Compact
    x_1 = tf.keras.layers.ZeroPadding2D(padding=((0, 0), (0, 1))) (x_1)
    x_1 = tf.keras.layers.Conv2D(24, 3, padding = 'same', activation = 'relu')(x_1)
    y = tf.keras.layers.Conv2D(1, 3, padding = 'same', activation = 'relu')(x_1)
    y = tf.keras.layers.AveragePooling2D(pool_size=(25, 25), strides=None, padding='valid', data_format=None)(y)
    y = tf.keras.layers.Flatten()(y)
    y = tf.keras.layers.Softmax()(y)
    
    CNNmodel2 = tf.keras.models.Model(input, y, name='segmentation_network')
    CNNmodel2.summary()

    adam = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    CNNmodel2.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics=['mae', 'acc'])

    return CNNmodel2
######

###### Get Reconnet model
def get_Reconnet(input_shape):

    input = tf.keras.layers.Input(shape = input_shape, name='input')
    x_0 = tf.keras.layers.Conv2D(64, 11, padding = 'same', activation='relu')(input)
    x_1 = tf.keras.layers.Conv2D(32, 1, padding = 'same', activation='relu')(x_0) # ?
    x_2 = tf.keras.layers.Conv2D(1, 7, padding = 'same', activation='relu')(x_1)
    x_3 = tf.keras.layers.Conv2D(64, 11, padding = 'same', activation='relu')(x_2)
    x_4 = tf.keras.layers.Conv2D(32, 1, padding = 'same', activation='relu')(x_3)
    y = tf.keras.layers.Conv2D(1, 7, padding = 'same', activation='relu')(x_4)
    
    y = tf.keras.layers.AveragePooling2D(pool_size=(25, 25), strides=None, padding='valid', data_format=None)(y)
    y = tf.keras.layers.Flatten()(y)
    y = tf.keras.layers.Softmax()(y)

    RECONmodel = tf.keras.models.Model(input, y, name='segmentation_network')

    adam = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    RECONmodel.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics=['mae', 'acc'])

    RECONmodel.summary()

    return RECONmodel
######