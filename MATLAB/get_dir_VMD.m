function get_dir_VMD(im,subject,method,task,rep,seg,chn,mode)
%Obtiene la dirección necesaria para crear una imagen según su
%sujeto, metodo, tarea, segmento y repetición para con ella crear la imagen en
%formato .jgp con dimension 128 x 128 
%
%   Parametros: im(matriz tiempo-frecuencia),sujeto,metodo,tarea,segmento y repetición
%   Output: ninguno 

   global localdir 
   
    switch task
           case 1
              task_str = ['\Baseline'];
           case 2
              task_str = ['\Mult'];
           case 3
              task_str = ['\Letter'];
           case 4
              task_str = ['\Rotation'];
           case 5
              task_str = ['\Counting'];
    end
   str = [localdir,'\Metodo_',num2str(method),'\sujeto',num2str(subject),task_str,'\suj',num2str(subject),'rep',num2str(rep),'chn',num2str(chn),'seg',num2str(seg),'m',num2str(mode),'.jpeg'];    
   imwrite(imresize(im,[128 128]),str);
end