function get_dir_aug(im,subject,method,task,rep,chn,mode)
%Obtiene la dirección necesaria para crear una imagen según su
%sujeto, metodo, tarea, repetición y modo para con ella crear la imagen en
%formato .jgp con dimension 128 x 128. Utiliza los residuos de cada modo
%para hacer data augmentation 
%
%   Parametros: im(matriz TF por SSWT),sujeto,metodo,tarea,repetición y
%   modo
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