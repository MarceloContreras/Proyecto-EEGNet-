# Proyecto-EEGNet-
Clasificador de cognitive tasks para señales EEG basado en EEGNet utilizando el dataset de Aunon y Keirn (1989). Las señales son preprocesadas mediante filtro Notch(60 Hz) y Eliptico(< 100 Hz)
con una normalización por z-score. Para generar imagenes analizables por una CNN, se combino el uso de mapas tiempo-frecuencia por CWT y la decomposición MVMD.

![alt text](https://github.com/MarceloContreras/Proyecto-EEGNet-/blob/main/Framework1.JPG)
![alt text](https://github.com/MarceloContreras/Proyecto-EEGNet-/blob/main/Framework2.JPG)

# Dataset utilizado 
Keirn & Aunon (1989) https://www.cs.colostate.edu/eeg/main/data/1989_Keirn_and_Aunon.

# Dependencias necesarias

## Python 
* Versión: 3.8.5
* Jupyter Notebook 6.1.4
* TensorFlow 2.6.0
* Numpy 1.19.2
* Scikit Learn 0.23.2
* Keras 2.6.0
* Keras processing 1.1.2

## Matlab 
* Versión: 9.8.0.1451342 R2020a Update 5 
* DSP System Toolbox Version 14.0 R2020a
* Image Processing Toolbox Version 11.1 R2020a
* Signal Processing Toolbox Version 8.4 R2020a
* Wavelet Toolbox Version 5.4 R2020a

# Ejecución

1. Definir la carpeta con su path donde se guardaran las imagenes generadas en main.m
2. Ejecutar main.m
3. Ordenar las carpetas de las imagenes generadas según las simulaciones que se desee realizar
4. Definir esta última carpeta en el archivo main.py
5. Ejecutar main.py

# Manejo de archivos

Ejemplo de carpetas para MATLAB

```
.
└── root/
    └── Metodo_1/
        ├── sujeto1/
        │   ├── Baseline/
        │   │   ├── sub1rep1ch1.jpeg
        │   │   ├── sub1rep1ch2.jpeg
        │   │   ├── sub1rep1ch3.jpeg
        │   │   └── ...
        │   ├── Counting/
        │   │   └── ...
        │   ├── Rotation/
        │   │   └── ...
        │   └── ...
        ├── sujeto2
        └── ...
```

Ejemplo de carpetas para Python

```
.
└── root/
    ├── Test1/
    │   ├── Baseline/
    │   │   ├── sub1rep1chn1.jpeg
    │   │   ├── sub1rep1chn2.jpeg
    │   │   └── ...
    │   └── Counting/
    │       ├── sub1rep1chn1.jpeg
    │       ├── sub1rep1chn2.jpeg
    │       └── ...
    ├── Test2/
    │   ├── Baseline/
    │   │   ├── sub1rep1chn1.jpeg
    │   │   ├── sub1rep1chn2.jpeg
    │   │   └── ...
    │   └── Rotation/
    │       └── ...
    └── ...
```
