import os 
import numpy as np
from model import *
from sklearn.model_selection import KFold,RepeatedKFold
from keras.preprocessing.image import ImageDataGenerator
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

def print_subject_results(loss_per_fold,acc_per_fold,repetitions):
    """
    Imprime los accuracy conseguido en cada Fold y calcula el acc y loss promedio de los folds de cada run
    
    Argumentos: Vector de loss dim(folds * repeticiones)
                Vector de accuracy 
    Output: Acc promedio final
            Std promedio final
    """
    
    # -- Averages scores 
    promedio_final = []
    print('------------------')
    print('Precision por fold')
    for i in range(0,len(acc_per_fold)):
        print('-----------------')
        print(f'Fold{i+1} - Loss: {loss_per_fold[i]} - Accuracy : {acc_per_fold[i]}')
    print('----------')
    for i in range(repetitions):
        promedio_final.append(np.mean(acc_per_fold[0*i:5*(i+1)]))
    #print('Precision promedio')
    #print(f'Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print('Precision Final')
    print(f'Accuracy: {np.mean(promedio_final)} (+- {np.std(promedio_final)})')
    
    return np.mean(promedio_final),np.std(promedio_final)



def Kcross_validation(num_class,inputs,targets,repetitions):
    """
    Aplica Repeated K-cross validation considerando la repeticiones deseadas
    
    Argumentos: num_class
                inputs(imagenes)
                targets(clases)
                repetitions
                
    Output: Acc promedio final
            Std promedio final
    """
    
    # Per-fold score containers 
    acc_per_fold = []
    loss_per_fold = []
    kfold = RepeatedKFold(n_splits = 5, n_repeats = repetitions)
    fold_n = 1
    
    for train,test in kfold.split(inputs,targets):
        EEGNet = get_EEGNet(num_class)
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_n} ...')

        history = EEGNet.fit(inputs[train] , targets[train], 
                            epochs = 50, steps_per_epoch = 2
                            )

        scores = EEGNet.evaluate(inputs[test],targets[test],verbose=0)
        print(f'Score for fold {fold_n}: {EEGNet.metrics_names[0]} of {scores[0]}; {EEGNet.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increse number of fold

        fold_n = fold_n + 1
    
    promedio,desviacion = print_subject_results(loss_per_fold,acc_per_fold,repetitions)
    
    return promedio,desviacion