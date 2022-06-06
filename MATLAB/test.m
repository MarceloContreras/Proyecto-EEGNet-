clc; clear all; close all;

%% Loading data 

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
 
x = preprocessing(1,1,1);
x = x(1,:);

%% VMD

imfBad = vmd(x,'NumIMFs',5);

% Show the estimated IF

modo1 = imfBad(:,1);
modo2 = imfBad(:,2);
modo3 = imfBad(:,3);
modo4 = imfBad(:,4);
modo5 = imfBad(:,5);

t = 0:1/Fs:10;
t = t(1:end-1);

figure(1),title("VMD")
subplot(2,3,1),plot(t,modo1),title("IMF_{1}");
subplot(2,3,2),plot(t,modo2),title("IMF_{2}");
subplot(2,3,3),plot(t,modo3),title("IMF_{3}");
subplot(2,3,4),plot(t,modo4),title("IMF_{4}");
subplot(2,3,5),plot(t,modo5),title("IMF_{5}");


