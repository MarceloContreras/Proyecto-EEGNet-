function makeImageRGBaugmentation(method)
%makeImageRGB: accede a la data necesaria de cada sample y canal,
%preprocesa la señal, aplica la transformada T-F elegida, utiliza una
%paleta RGB y crea las imagenes. En total se crean 120 muestras (10 repeticiones x 6 canales x 2 tareas)
%de imagenes 128 x 128 x 3
%
%   Parametros: el num de metodo
%   Output: ninguno
    
    global Fs
    
    for subject = 1:4
        for task = 1:5
            if subject == 2
               for rep = 1:5 % El sujeto 2 solo tiene 5 repeticiones
                    x = preprocessing(subject,task,rep);
                    [modes, u_hat, omega] = MVMD_new(x, 2000, 0, 4, 0, 1, 1e-7); %Decomposicion multi canal
                    y = sum(modes,1); % Suma de modos(IMFs) eliminando el residual
                    for chn=1:6
                        cfs = wsst(y(:,:,chn),Fs,'bump','VoicesPerOctave',12); % Transformada T-F seleccionada
                        im = ind2rgb(im2uint8(rescale(abs(cfs))),jet(128)); % Paleta RGB seleccionada
                        dirAugImage(im,subject,method,task,rep,chn,0) %% Esta funcion debe cambiarse para aceptar Testing o Training
                        for nmode = 1:4
                           residues = x(chn,:) - modes(nmode,:,chn); 
                           cfs = wsst(residues,Fs,'bump','VoicesPerOctave',12); % Transformada T-F seleccionada
                           im = ind2rgb(im2uint8(rescale(abs(cfs))),jet(128)); % Paleta RGB seleccionada
                           dirAugImage(im,subject,method,task,rep,chn,nmode) %% Esta funcion debe cambiarse para aceptar Testing o Training
                        end
                    end
               end
            else
                for rep = 1:10 % Los demás sujetos (1,3,4) tienen 10 repeticiones
                    x = preprocessing(subject,task,rep);
                    [modes, u_hat, omega] = MVMD_new(x, 2000, 0, 4, 0, 1, 1e-7);
                    y = sum(modes,1);
                    for chn=1:6
                        cfs =wsst(y(:,:,chn),Fs,'bump','VoicesPerOctave',12);
                        im = ind2rgb(im2uint8(rescale(abs(cfs))),jet(128));
                        dirAugImage(im,subject,method,task,rep,chn,0)
                        for nmode = 1:4
                           residues = x(chn,:) - modes(nmode,:,chn); 
                           cfs = wsst(residues,Fs,'bump','VoicesPerOctave',12); % Transformada T-F seleccionada
                           im = ind2rgb(im2uint8(rescale(abs(cfs))),jet(128)); % Paleta RGB seleccionada
                           dirAugImage(im,subject,method,task,rep,chn,nmode) %% Esta funcion debe cambiarse para aceptar Testing o Training
                        end
                    end
               end  
            end
        end
    end
end
