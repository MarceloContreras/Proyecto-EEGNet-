import tensorflow as tf
from tensorflow.keras import Input, Model, optimizers,regularizers
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Dense,\
                                    Conv2D,\
                                    BatchNormalization,\
                                    AveragePooling2D, \
                                    MaxPooling2D, \
                                    DepthwiseConv2D, \
                                    Activation, \
                                    Dropout,\
                                    Flatten
def EEGNet_model(num_class):
    """
    Construye una EEGNet utilizando Tensorflow como objeto secuencial.
    
    Argumentos: num_class
    Output: EEGNet como modelo de Tensorflow.keras
    """
    
    EEGNet = tf.keras.Sequential()

    # Block1
    regularizers.l2(1e-4)
    EEGNet.add(Conv2D(4, (1, 125),
                        padding='same',
                        use_bias=False,
                        name='tfconv',input_shape = (128,128,3)))
    EEGNet.add(BatchNormalization(axis=-1))
    EEGNet.add(DepthwiseConv2D((6, 1),
                             use_bias=False,
                             depth_multiplier=2,
                             depthwise_constraint=max_norm(1.),
                             name='sconv'))
    EEGNet.add(BatchNormalization(axis=-1))
    EEGNet.add(Activation('elu'))
    EEGNet.add(AveragePooling2D((1, 4)))
    EEGNet.add(Dropout(0.5))

    # Block 2

    EEGNet.add(Conv2D(8, (1, 32),
                             padding='same',
                             use_bias=False,
                             name='fs',
                             kernel_regularizer='l2'
                     ))
    EEGNet.add(BatchNormalization(axis=-1))
    EEGNet.add(Activation('elu'))
    EEGNet.add(AveragePooling2D((1, 8)))
    EEGNet.add(Dropout(0.5))

    # Output

    EEGNet.add(Flatten(name='flatten'))

    EEGNet.add(Dense(num_class,
                  name='dense',
                  kernel_constraint=max_norm(0.25)))
    EEGNet.add(Activation('softmax', name='softmax'))

    return EEGNet



def get_compile(model: tf.keras.Model):
    """
    Compila el modelo con un optimizador Adam (lr = 0.001), loss categorico y como metrica el accuracy
    
    Argumentos: CNN como modelo
    Output: Modelo compilado
    """
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                   loss='sparse_categorical_crossentropy',
                   metrics=['acc'])
    return model



def get_EEGNet(num_class):
    """
    Crea y compila una EEGNet lista para entrenarla y clasificar. Se adapta según la 
    cantidad de clases
    
    Argumentos: num_class
    Output: modelo listo 
    """
    model = EEGNet_model(num_class)
    model = get_compile(model)
    
    return model