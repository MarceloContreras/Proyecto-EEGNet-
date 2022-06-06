import os 
import numpy as np
import matplotlib.pyplot as plt
from EEGNet_model import *
from sklearn.model_selection import KFold,RepeatedKFold,train_test_split, LeavePOut
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
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


def plot_acc(history,fold):
    plt.title('Train Accuracy vs Val Accuracy Fold:' + str(fold))
    plt.plot(history.history['acc'], label='Train Accuracy Fold ', color='black')
    plt.plot(history.history['val_acc'], label='Val Accuracy Fold ', color='red', linestyle = "dashdot")
    plt.legend()
    plt.show()
    
def plot_loss(history,fold):
    plt.title('Train Loss vs Val Loss Fold:' + str(fold))
    plt.plot(history.history['loss'], label='Train Loss Fold ', color='black')
    plt.plot(history.history['val_loss'], label='Val Loss Fold ', color='red', linestyle = "dashdot")
    plt.legend()
    plt.show()
    
def plot_metrics(history,fold):
    plot_acc(history,fold)
    plot_loss(history,fold)

def print_subject_results(val_per_fold,acc_per_fold,repetitions):
    """
    Imprime los accuracy conseguido en cada Fold y calcula el acc y loss promedio de los folds de cada run
    
    Argumentos: Vector de loss dim(folds * repeticiones)
                Vector de accuracy 
    Output: Acc promedio final
            Std promedio final
    """
    
    test_score_final = []
    val_score_final = []
    print('******************')
    print('Precision por run')
    print('**************************')
    print('*Val_acc --------Test_acc*')
    for i in range(0,repetitions):
        max_per_fold = max(acc_per_fold[0*i:5*(i+1)])
        test_score_final.append(max_per_fold[1])
        val_score_final.append(np.mean(val_per_fold[0*i:5*(i+1)]))

    val_mean = np.mean(val_score_final)    
    val_std = np.std(val_score_final)
    test_mean = np.mean(test_score_final)
    test_std = np.std(test_score_final)
        
    print(f'Val Accuracy:{val_mean*100} +- {val_std*100}')
    print(f'Test Accuracy:{test_mean*100} +- {test_std*100}')
    
    return val_mean,val_std,test_mean,test_std





def Kcross_validation_v1(num_class,X_train,X_test,Y_train,Y_test,repetitions):
    """
    Aplica Repeated K-cross validation considerando la repeticiones deseadas
    
    Argumentos: num_class
                inputs(imagenes)
                targets(clases)
                repetitions
                
    Output: val_mean,val_std,test_mean,test_std
    """
    
    # Per-fold score containers 
    acc_per_fold = []
    val_per_fold = []
    kfold = RepeatedKFold(n_splits = 5, n_repeats = repetitions)
    fold_n = 1
    
    for train,val in kfold.split(X_train,Y_train):
        EEGNet = get_EEGNet(num_class)
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_n} ...')

        pat = 25
        early_stopping = EarlyStopping(monitor='val_acc', mode='max', patience=pat, verbose=1)
        model_checkpoint = ModelCheckpoint('./model_checkpoint', verbose=1, save_best_only=True, monitor='val_acc',
        mode='max')
        
        history = EEGNet.fit(X_train[train] , Y_train[train], 
                             epochs = 50, steps_per_epoch = 2, validation_data = (X_train[val],Y_train[val]), 
                             callbacks=[early_stopping, model_checkpoint]
                            )
        plot_metrics(history,fold_n)
    
        EGGNetnew = tf.keras.models.load_model('./model_checkpoint')
        val_scores = EGGNetnew.evaluate(X_train[val],Y_train[val],verbose=2)
        test_scores = EGGNetnew.evaluate(X_test,Y_test,verbose=2)

        print(f'Val-Score for fold {fold_n}: {EEGNet.metrics_names[0]} of {val_scores[0]}; {EEGNet.metrics_names[1]} of {val_scores[1]*100}%')
        print(f'Test-Score for fold {fold_n}: {EEGNet.metrics_names[0]} of {test_scores[0]}; {EEGNet.metrics_names[1]} of {test_scores[1]*100}%')
        acc_per_fold.append((val_scores[1],test_scores[1]))
        val_per_fold.append(val_scores[1])

        # Increse number of fold

        fold_n = fold_n + 1
    
    val_mean,val_std,test_mean,test_std = print_subject_results(val_per_fold,acc_per_fold,repetitions)
    
    return val_mean,val_std,test_mean,test_std





def Kcross_validation_v2(num_class,X_train,X_test,Y_train,Y_test,repetitions):

    """
    Aplica Repeated K-cross validation considerando la repeticiones deseadas. Separa training y validation por canales 
    donde se valida con 1 canal. El canal utilizado en testeo se obtiene ya desde get_dataset
    
    Argumentos: num_class
                X_train,Y_train
                X_test,Y_train
                repetitions
                
    Output: val_mean,val_std,test_mean,test_std
        
    """
    
    # Per-fold score containers 
    acc_per_fold = []
    val_per_fold = []
    kfold = RepeatedKFold(n_splits = 5, n_repeats = repetitions)
    
    fold_n = 1
    list_channels = np.array([0,1,2,3,4])
    step_per_chn = 50
    
    for train,val in kfold.split(list_channels):
        
        train_index = np.zeros(shape = 0)
        val_index = np.zeros(shape = 0)

        for i in train:
            temp = range(i*step_per_chn,(i+1)*step_per_chn)
            train_index = np.concatenate((train_index,temp))

        for k in val:
            temp = range(k*step_per_chn,(k+1)*step_per_chn)
            val_index = np.concatenate((val_index,temp))
        
        train_index = np.concatenate((train_index,train_index+250,train_index+500))
        val_index = np.concatenate((val_index,val_index+250,val_index+500))

        np.random.shuffle(train_index)
        np.random.shuffle(val_index)
        
        train_index = train_index.astype(int)
        val_index = val_index.astype(int)
        
        EEGNet = get_EEGNet(num_class)
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_n} and Validating with Chn {val[0]+1}')

        pat = 50
        early_stopping = EarlyStopping(monitor='val_acc', mode='max', patience=pat, verbose=1)
        model_checkpoint = ModelCheckpoint('./model_checkpoint', verbose=1, save_best_only=True, monitor='val_acc',
        mode='max')
        
        history = EEGNet.fit(X_train[train_index] , Y_train[train_index], 
                             epochs = 100, steps_per_epoch = 5, validation_data = (X_train[val_index],Y_train[val_index]), 
                             callbacks=[early_stopping, model_checkpoint]
                            )
        plot_metrics(history,fold_n)
    
        EGGNetnew = tf.keras.models.load_model('./model_checkpoint')
        val_scores = EGGNetnew.evaluate(X_train[val_index],Y_train[val_index],verbose=0)
        test_scores = EGGNetnew.evaluate(X_test,Y_test,verbose=0)
    
        print(f'Val-Score for fold {fold_n}: {EEGNet.metrics_names[0]} of {val_scores[0]}; {EEGNet.metrics_names[1]} of {val_scores[1]*100}%')
        print(f'Test-Score for fold {fold_n}: {EEGNet.metrics_names[0]} of {test_scores[0]}; {EEGNet.metrics_names[1]} of {test_scores[1]*100}%')
        acc_per_fold.append((val_scores[1],test_scores[1]))
        val_per_fold.append(val_scores[1])
        
         # Increse number of fold
        fold_n = fold_n + 1
        
    val_mean,val_std,test_mean,test_std = print_subject_results(val_per_fold,acc_per_fold,repetitions)
    
    return val_mean,val_std,test_mean,test_std


# Cargo mi data
# Sujeto 1 ~ Data de todas las permutaciones




def Kcross_validation_v3(num_class,X_train,X_val,X_test,Y_train,Y_val,Y_test,repetitions):
    """
    Aplica Leave-One-Out para hacer las distintas permutaciones necesarias en el test de 
    independencia. Los diferentes datasets fueron separados previamente 
    
    Argumentos: num_class
                X_train,Y_train
                X_val,Y_val
                X_test,Y_train
                repetitions
                
    Output: val_mean,val_std,test_mean,test_std
        
    """
    
    # Per-fold score containers 
    acc_per_fold = []
    val_per_fold = []
    lpo = LeavePOut(p=2)
    
    fold_n = 1
    list_channels = np.array([0,1,2,3,4])
    step_per_chn_train = 6
    step_per_chn_val = 6
    
    for train,val in lpo.split(list_channels):
        
        train_index = np.zeros(shape = 0)
        val_index = np.zeros(shape = 0)

        for i in train:
            temp = range(i*step_per_chn_train,(i+1)*step_per_chn_train)
            train_index = np.concatenate((train_index,temp))

        for k in val:
            temp = range(k*step_per_chn_val,(k+1)*step_per_chn_val)
            val_index = np.concatenate((val_index,temp))
        
        train_index = np.concatenate((train_index,train_index+30))
        val_index = np.concatenate((val_index,val_index+30))
        
        train_index = train_index.astype(int)
        val_index = val_index.astype(int)
        
        np.random.shuffle(train_index)
        np.random.shuffle(val_index)
        
        EEGNet = get_EEGNet(num_class)
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_n} and Validating with Chn {val[0]+1}')
        print(f'Entrenando con canales {train} y validando con canales {val}')

        pat = 50
        early_stopping = EarlyStopping(monitor='val_acc', mode='max', patience=pat, verbose=1)
        model_checkpoint = ModelCheckpoint('./model_checkpoint', verbose=1, save_best_only=True, monitor='val_acc',
        mode='max')
        
        history = EEGNet.fit(X_train[train_index],Y_train[train_index],
                             epochs = 50, steps_per_epoch = 2, validation_data = (X_val[val_index],Y_val[val_index]), 
                             callbacks=[early_stopping, model_checkpoint]
                            )
        plot_metrics(history,fold_n)
    
        EGGNetnew = tf.keras.models.load_model('./model_checkpoint')
        val_scores = EGGNetnew.evaluate(X_val[val_index],Y_val[val_index],verbose=2)
        test_scores = EGGNetnew.evaluate(X_test,Y_test,verbose=2)
    
        print(f'Val-Score for fold {fold_n}: {EEGNet.metrics_names[0]} of {val_scores[0]}; {EEGNet.metrics_names[1]} of {val_scores[1]*100}%')
        print(f'Test-Score for fold {fold_n}: {EEGNet.metrics_names[0]} of {test_scores[0]}; {EEGNet.metrics_names[1]} of {test_scores[1]*100}%')
        acc_per_fold.append((val_scores[1],test_scores[1]))
        val_per_fold.append(val_scores[1])
        
         # Increse number of fold
        fold_n = fold_n + 1
        
    val_mean,val_std,test_mean,test_std = print_subject_results(val_per_fold,acc_per_fold,repetitions)
    
    return val_mean,val_std,test_mean,test_std