from model import *
from dataset import *
from training import *

resultados_finales = []
num_test = 9
num_class = 3
repetitions = 1

for i in range(num_test):
    dir_task = r'C:\Users\Lenovo\Documents\UTEC\Ciclo 7\ProyectoCNN\Python\Dataset\Test' + str(i+1)
    inputs,targets = get_dataset(dir_task,num_class)
    mean,std = Kcross_validation(num_class,inputs,targets,repetitions)
    resultados_finales.append(f'{mean} +- {std}')