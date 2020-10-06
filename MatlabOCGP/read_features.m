% Description:  Reads data from the Drug Target dataset and log-trasform
% heavy tailed features
%                      
% Author:       Antonio De Falco           
%  

function x=read_features(T,features,features_log)
    x = []
    
    for i = features
        if isnumeric(table2array(T(:,i)))
            v = table2array(T(:,i));   
        elseif iscellstr(table2array(T(:,i)))
            v = str2double(table2array(T(:,i)));
        end
        if ismember(i,features_log)
            v = log(v + 0.01);
        end    
        
    x=[x,v];
    end
   
end