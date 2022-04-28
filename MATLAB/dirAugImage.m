function dirAugImage(im,subject,method,task,rep,chn,mode)
%dirImage: Obtiene la dirección necesaria para crear una imagen según su
%sujeto, metodo, tarea y repetición para con ella crear la imagen en
%formato .jgp con dimension 128 x 128 
%
%   Parametros: im(matriz tiempo-frecuencia de la transformada utilizada Ex.STFT,CWT,S-transform),sujeto,metodo,tarea y repetición
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
   str = [localdir,'\Metodo_',num2str(method),'\sujeto',num2str(subject),task_str,'\suj',num2str(subject),'chn',num2str(chn),'rep',num2str(rep),'m',num2str(mode),'.jpeg'];
   imwrite(imresize(im,[128 128]),str);
end