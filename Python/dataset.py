import os 
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def get_dataset(dir_task,num_class):
    """
    Construye el dataset a partir de la direccion de carpetas y la cantidad de clases
    a clasificar.
    
    Argumentos: dir_task(Direccion de carpetas) y num_class(cantidad de clases)
    Output: inputs(imagenes) y targets(vector de clases) 
    """
    
    type_class = 'binary'
    if (num_class != 2):
        type_class = 'sparse'
    
    datagen_task = ImageDataGenerator(rescale=1./255)

    data_task = datagen_task.flow_from_directory(
        dir_task, 
        batch_size = 100,
        target_size=(128, 128),
        class_mode = type_class
        )
    
    num_samples = 0
    for i in range(len(data_task)):
        num_samples += len(data_task[i][1]) 


    inputs = np.zeros(shape=(num_samples, 128, 128, 3))
    targets = np.zeros(shape=(num_samples))
    i=0

    for inputs_batch,labels_batch in data_task:
        inputs[i * 100 : (i + 1) * 100] =  inputs_batch
        targets[i * 100 : (i + 1) * 100] = labels_batch
        i += 1
        if i * 100 >= num_samples:
            break
    
    return inputs,targets