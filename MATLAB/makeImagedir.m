function makeImagedir(method)
%makeImagedir: Crea las carpetas necesarios para realizar pruebas en CNN.
%Presenta la siguiente estructura:
%   \local dir
%   ...\Metodo_n
%       ...\sujeto_n
%           ...\Tarea_1
%           ...\Tarea_2
%           ... ~~
%
%   Parametros: el # de metodo
%   Output: ninguno

    global localdir
    
    tasks = ["\Baseline" "\Mult" "\Letter" "\Rotation" "\Counting"];
    for subject = 1:4
        for i = 1:5
            Folder = [localdir,'\Metodo_',num2str(method),'\sujeto',num2str(subject),convertStringsToChars(tasks(i))];
            if not(isfolder(Folder)) %Comprueba que el folder exista, en caso opuesto lo crea
                mkdir(Folder)
            end
        end
    end
end
    