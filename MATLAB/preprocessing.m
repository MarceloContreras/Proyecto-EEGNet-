function x = preprocessing(subject,task,rep)
%Preproccesing: Aplica filtro Notch en 60 Hz, Filtro Eliptico < 100Hz y zscore a una señal EEG
%
%   Parametros: sujeto, tarea y repetición
%   Output: Señal EEG prepocesada ~ dim(2500,1)
%
    global data A B z
    
    y = canalesdata(subject,task,rep);
    y = y(1:6,:);
    y_filtered = [];
    for chn = 1:6
        y_filtered(chn,:) = filtfilt(z,double(y(chn,:))); %Filtro Notch 
        y_filtered(chn,:) = filtfilt(B,A,y_filtered(chn,:)); %Filtro Eliptic   
    end
    x = normalize(y_filtered,2); % Z-score
end