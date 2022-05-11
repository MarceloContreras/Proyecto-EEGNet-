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

imfBad = vmd(x,'NumIMFs',4);

%% Show the estimated IF

modo1 = imfBad(:,1);
modo2 = imfBad(:,2);
modo3 = imfBad(:,3);
modo4 = imfBad(:,4);

figure(1)
subplot(2,2,1),spectrogram(modo1,hamming(120),[],[],Fs,'yaxis'),title("IMF_{1}");
subplot(2,2,2),spectrogram(modo2,hamming(120),[],[],Fs,'yaxis'),title("IMF_{2}");
subplot(2,2,3),spectrogram(modo3,hamming(120),[],[],Fs,'yaxis'),title("IMF_{3}");
subplot(2,2,4),spectrogram(modo4,hamming(120),[],[],Fs,'yaxis'),title("IMF_{4}");
%% EMD

imfBad = emd(x,'MaxNumIMF',4);

%% Show the estimated IF

modo1 = imfBad(:,1);
modo2 = imfBad(:,2);
modo3 = imfBad(:,3);
modo4 = imfBad(:,4);

figure(2),title("EMD")
subplot(2,2,1),spectrogram(modo1,hamming(120),[],[],Fs,'yaxis'),title("IMF_{1}");
subplot(2,2,2),spectrogram(modo2,hamming(120),[],[],Fs,'yaxis'),title("IMF_{2}");
subplot(2,2,3),spectrogram(modo3,hamming(120),[],[],Fs,'yaxis'),title("IMF_{3}");
subplot(2,2,4),spectrogram(modo4,hamming(120),[],[],Fs,'yaxis'),title("IMF_{4}");

segments = reshape(modo1,250,[]);
