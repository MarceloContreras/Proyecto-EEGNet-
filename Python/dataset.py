import os 
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image


def get_dataset_v1(dir_task,num_class):
    """
    Construye el dataset a partir de la direccion de carpetas y la cantidad de clases
    a clasificar.
    
    Argumentos: dir_task(Direccion de carpetas) y num_class(cantidad de clases)
    Output: inputs(imagenes) y targets(vector de clases) 
    """
    
    BATCH_SIZE = 100

    type_class = 'binary'
    if (num_class != 2):
        type_class = 'sparse'
    
    datagen_task = ImageDataGenerator(rescale=1./255)

    data_task = datagen_task.flow_from_directory(
        dir_task, 
        batch_size = BATCH_SIZE,
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
        inputs[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] =  inputs_batch
        targets[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] = labels_batch
        i += 1
        if i * BATCH_SIZE >= num_samples:
            break
    
    return inputs,targets




def get_dataset_v2(dir_task,num_class):
    """
    Construye el dataset a partir de la direccion de carpetas y la cantidad de clases
    a clasificar. La v2 get_dataset obtiene las imagenes de las tres clases de forma ordenada por repeticiones/canales
    
    Argumentos: dir_task(Direccion de carpetas) y num_class(cantidad de clases)
    Output: inputs(imagenes) y targets(vector de clases) 
    """
    #Loading
    train_dir = os.path.join(dir_task, "train_data")
    test_dir = os.path.join(dir_task, "test_data")
    
    #Parameteres
    TRAIN_SIZE = 250
    BATCH_SIZE = 100

    # Training
    X_train1 = np.zeros(shape=(TRAIN_SIZE,128, 128, 3))
    X_train2 = np.zeros(shape=(TRAIN_SIZE,128, 128, 3))
    X_train3 = np.zeros(shape=(TRAIN_SIZE,128, 128, 3))
    
    fnames_tasks = [os.path.join(train_dir,fname) for fname in os.listdir(train_dir)]
    
    fnames = [os.path.join(fnames_tasks[0],fname) for fname in os.listdir(fnames_tasks[0])]
    for i in range(0,TRAIN_SIZE):
        img_path = fnames[i]
        img = image.load_img(img_path,target_size=(128,128))
        x = image.img_to_array(img)
        x = x.astype('float32') / 255
        X_train1[i] = x
    
    fnames = [os.path.join(fnames_tasks[1],fname) for fname in os.listdir(fnames_tasks[1])]
    for i in range(0,TRAIN_SIZE):
        img_path = fnames[i]
        img = image.load_img(img_path,target_size=(128,128))
        x = image.img_to_array(img)
        x = x.astype('float32') / 255
        X_train2[i] = x
    
    fnames = [os.path.join(fnames_tasks[2],fname) for fname in os.listdir(fnames_tasks[2])]
    for i in range(0,TRAIN_SIZE):
        img_path = fnames[i]
        img = image.load_img(img_path,target_size=(128,128))
        x = image.img_to_array(img)
        x = x.astype('float32') / 255
        X_train3[i] = x
   
    X_train = np.concatenate((X_train1,X_train2,X_train3),axis = 0)
    
    Y_task1 = np.zeros((TRAIN_SIZE,), dtype=np.float64)
    Y_task2 = np.ones((TRAIN_SIZE,), dtype=np.float64)
    Y_task3 = 2*np.ones((TRAIN_SIZE,), dtype=np.float64)
    
    Y_train = np.concatenate((Y_task1,Y_task2,Y_task3))
    
    # Testing
    
    type_class = 'binary'
    if (num_class != 2):
        type_class = 'sparse'
    
    datagen_task = ImageDataGenerator(rescale=1./255)
    
    test_task = datagen_task.flow_from_directory(
        test_dir, 
        batch_size = BATCH_SIZE,
        target_size=(128, 128),
        class_mode = type_class
        )
    
    num_samples = 0
    for i in range(len(test_task)):
        num_samples += len(test_task[i][1]) 
    
    X_test = np.zeros(shape=(num_samples, 128, 128, 3))
    Y_test = np.zeros(shape=(num_samples))
    i=0

    for inputs_batch,labels_batch in test_task:
        X_test[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] =  inputs_batch
        Y_test[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] = labels_batch
        i += 1
        if i * BATCH_SIZE >= num_samples:
            break
            
    return X_train,X_test,Y_train,Y_test





def get_dataset_v3(dir_task,num_class):
    """
    Construye el dataset a partir de la direccion de carpetas y la cantidad de clases
    a clasificar. La v3 get_dataset obtiene el train_set, val_set y test_set de forma separada
    para el test de independencia de clases
    
    Argumentos: dir_task(Direccion de carpetas) y num_class(cantidad de clases)
    Output: inputs(imagenes) y targets(vector de clases) 
   
    """
    
    train_dir = os.path.join(dir_task, "train_data")
    val_dir = os.path.join(dir_task, "val_data")
    test_dir = os.path.join(dir_task, "test_data")
    
    # Sizes
    TRAIN_SIZE = 30
    VAL_SIZE = 30
    BATCH_SIZE = 100

    # Training
    X_train1 = np.zeros(shape=(TRAIN_SIZE,128, 128, 3))
    X_train2 = np.zeros(shape=(TRAIN_SIZE,128, 128, 3))
    
    fnames_tasks = [os.path.join(train_dir,fname) for fname in os.listdir(train_dir)]
    
    fnames = [os.path.join(fnames_tasks[0],fname) for fname in os.listdir(fnames_tasks[0])]
    for i in range(0,TRAIN_SIZE):
        img_path = fnames[i]
        img = image.load_img(img_path,target_size=(128,128))
        x = image.img_to_array(img)
        x = x.astype('float32') / 255
        X_train1[i] = x
    
    fnames = [os.path.join(fnames_tasks[1],fname) for fname in os.listdir(fnames_tasks[1])]
    for i in range(0,TRAIN_SIZE):
        img_path = fnames[i]
        img = image.load_img(img_path,target_size=(128,128))
        x = image.img_to_array(img)
        x = x.astype('float32') / 255
        X_train2[i] = x
    
    X_train = np.concatenate((X_train1,X_train2),axis = 0)
    
    Y_train1 = np.zeros((TRAIN_SIZE,), dtype=np.float64)
    Y_train2 = np.ones((TRAIN_SIZE,), dtype=np.float64)
    Y_train = np.concatenate((Y_train1,Y_train2))

    # Validation 
    
    X_val1 = np.zeros(shape=(VAL_SIZE,128, 128, 3))
    X_val2 = np.zeros(shape=(VAL_SIZE,128, 128, 3))
    
    fnames_tasks = [os.path.join(val_dir,fname) for fname in os.listdir(train_dir)]
    
    fnames = [os.path.join(fnames_tasks[0],fname) for fname in os.listdir(fnames_tasks[0])]
    for i in range(0,VAL_SIZE):
        img_path = fnames[i]
        img = image.load_img(img_path,target_size=(128,128))
        x = image.img_to_array(img)
        x = x.astype('float32') / 255
        X_val1[i] = x
    
    fnames = [os.path.join(fnames_tasks[1],fname) for fname in os.listdir(fnames_tasks[1])]
    for i in range(0,VAL_SIZE):
        img_path = fnames[i]
        img = image.load_img(img_path,target_size=(128,128))
        x = image.img_to_array(img)
        x = x.astype('float32') / 255
        X_val2[i] = x

    X_val = np.concatenate((X_val1,X_val2),axis = 0)
    
    Y_val1 = np.zeros((VAL_SIZE,), dtype=np.float64)
    Y_val2 = np.ones((VAL_SIZE,), dtype=np.float64)
    Y_val = np.concatenate((Y_val1,Y_val2))

    #Testing 
    
    type_class = 'binary'
    if (num_class != 2):
        type_class = 'sparse'
    
    datagen_task = ImageDataGenerator(rescale=1./255)
    
    test_task = datagen_task.flow_from_directory(
        test_dir, 
        batch_size = BATCH_SIZE,
        target_size=(128, 128),
        class_mode = type_class
        )
    
    num_samples = 0
    for i in range(len(test_task)):
        num_samples += len(test_task[i][1]) 
    
    X_test = np.zeros(shape=(num_samples, 128, 128, 3))
    Y_test = np.zeros(shape=(num_samples))
    i=0

    for inputs_batch,labels_batch in test_task:
        X_test[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] =  inputs_batch
        Y_test[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] = labels_batch
        i += 1
        if i * BATCH_SIZE >= num_samples:
            break
            
    return X_train,X_val,X_test,Y_train,Y_val,Y_test