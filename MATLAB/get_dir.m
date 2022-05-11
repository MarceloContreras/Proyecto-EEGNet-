function get_dir(im,subject,method,task,rep,chn)
%Obtiene la dirección necesaria para crear una imagen según su
%sujeto, metodo, tarea y repetición para con ella crear la imagen en
%formato .jgp con dimension 128 x 128 
%
%   Parametros: im(matriz tiempo-frecuencia de la transformada utilizada Ex.STFT,CWT,S-transform),sujeto,metodo,tarea y repetición
%   Output: ninguno 

   global localdir 
   
   switch task
       case 1
          str = [localdir,'\Metodo_',num2str(method),'\sujeto',num2str(subject),'\Baseline','\suj',num2str(subject),'chn',num2str(chn),'rep',num2str(rep),'.jpeg'];
       case 2
          str = [localdir,'\Metodo_',num2str(method),'\sujeto',num2str(subject),'\Mult','\suj',num2str(subject),'chn',num2str(chn),'rep',num2str(rep),'.jpeg'];
       case 3
          str = [localdir,'\Metodo_',num2str(method),'\sujeto',num2str(subject),'\Letter','\suj',num2str(subject),'chn',num2str(chn),'rep',num2str(rep),'.jpeg'];
       case 4
          str = [localdir,'\Metodo_',num2str(method),'\sujeto',num2str(subject),'\Rotation','\suj',num2str(subject),'chn',num2str(chn),'rep',num2str(rep),'.jpeg'];
       case 5
          str = [localdir,'\Metodo_',num2str(method),'\sujeto',num2str(subject),'\Counting','\suj',num2str(subject),'chn',num2str(chn),'rep',num2str(rep),'.jpeg'];
   end     
   imwrite(imresize(im,[128 128]),str);
end


