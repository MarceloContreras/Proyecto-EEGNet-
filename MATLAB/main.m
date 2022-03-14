clc;
clear all;
close all;

%% Cargando data

% Listado de tareas: 1. Baseline 2.Mult 3.Letter 4.Rotation 5.Counting
% Channels: 1 al 6
% Dato incompleto: Sujeto 4- Tarea 3 Letter Celda 195 Repeticion 10

global data Fs
load('eegdata.mat','data')
Fs=250;
samples=2500;
% Segmentación
segment=500; 
overlap=100; 

%% Directorio base a elegir 

global localdir

localdir = 'C:\Users\Lenovo\Documents\UTEC\Ciclo 7\ProyectoCNN\MetodoPrueba';

%% Preprocesamiento 

% Filtro Notch para 60 Hz

wo = 60/(Fs/2);  
bw = wo/10;
[D,C] = iirnotch(wo,bw);

% Filtro Notch para 60 Hz

global z 
z = designfilt('bandstopiir','FilterOrder',2, ...
               'HalfPowerFrequency1',59,'HalfPowerFrequency2',61, ...
               'DesignMethod','butter','SampleRate',Fs);

% Filtro pasa bajos de 0-100 Hz
global B A

Rp=0.5;
Rs = 30;
orden=7;
[B, A] = ellip(orden,Rp,Rs,100/(Fs/2));

%% MVMD, transformada y generación de imagenes 

method = 1;

% makeImagedir(method);
% makeImageRGB(method);

a = imread('suj1rep1chn1.jpeg');
b = imread('suj1rep1chn12.jpeg'); 
result = all(size(a) == size(b));
if result
    result = all(reshape(a,[],1)== reshape(b,[],1));
end





