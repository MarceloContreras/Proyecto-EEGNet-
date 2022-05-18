function get_dataset_VMD(method)
%Accede a la data necesaria de cada sample y canal,
%preprocesa la señal, segmente la data, aplica VMD,aplica la transformada T-F elegida, utiliza una
%paleta RGB y crea las imagenes. En total se crean 1200 muestras (10 repeticiones x 6 canales x 2 tareas x 10 segmentos)
%de imagenes 128 x 128 x 3
%
%   Parametros: el num de metodo
%   Output: ninguno


 global Fs
    
    for subject = 1:4
        for task = 1:5
            
            if subject == 2
               rep_i = 1:5;
            else
               rep_i = 1:10;
            end 

            for rep = rep_i % El sujeto 2 solo tiene 5 repeticiones
                x = preprocessing(subject,task,rep);
                for chn=1:6
                    segments = reshape(x(chn,:),500,[]);
                    for seg = 1:5
                        modes = vmd(segments(:,seg),"NumIMFs",4);
                        y = sum(modes,2);
                        cfs = wsst(y,Fs,'bump','VoicesPerOctave',12); % Transformada T-F seleccionada
                        im = ind2rgb(im2uint8(rescale(abs(cfs))),jet(128)); % Paleta RGB seleccionada
                        get_dir_VMD(im,subject,method,task,rep,seg,chn)
                    end
                end
            end
        end
    end
end
