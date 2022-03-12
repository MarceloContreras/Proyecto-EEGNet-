function z = canalesdata(sujeto,tarea,rep)
%canalesdata: Obtiene una repetición de sujeto~tarea considerando de los
%canales 1 a 6 
%
%   Parametros: sujeto, tarea y repetición
%   Output: señales EEG de duración total (2500 x 6)

    global data

    y= [];
    if(sujeto==1)
        if (rep>=1 && rep<=5)
            y=data{rep+5*(tarea-1)+(sujeto-1)*50}{4};
        elseif (rep>=6 && rep<=10)
            y=data{rep+20+5*(tarea-1)+(sujeto-1)*50}{4};
        end
    elseif(sujeto==2)
        y=data{rep+5*(tarea-1)+(sujeto-1)*50}{4};
    elseif(sujeto>=3 && sujeto<=4)
        if (rep>=1 && rep<=5)
        y=data{rep-25+5*(tarea-1)+(sujeto-1)*50}{4};
        elseif (rep>=6 && rep<=10)
        y=data{rep-5+5*(tarea-1)+(sujeto-1)*50}{4};
        end
    else
        if (rep>=1 && rep<=5)
        y=data{rep-25+5*(tarea-1)+(sujeto-1)*50}{4};
        elseif (rep>=6 && rep<=10)
        y=data{rep-5+5*(tarea-1)+(sujeto-1)*50}{4};
        elseif (rep>=11 && rep<=15)
        y=data{rep+15+5*(tarea-1)+(sujeto-1)*50}{4};    
        end
    end
    z=y;
end