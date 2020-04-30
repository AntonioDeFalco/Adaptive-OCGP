function [K,Ks,Kss]=se_kernel(svar,ls,x,y,dist)

    if strcmp(dist,'euclidean')
        if size(ls,1) == 1
            K   = svar*exp(-0.5*euclidean_distance(x,x)/ls);
            Ks = svar*exp(-0.5*euclidean_distance(x,y)/ls); 
        end

        if size(ls,1) > 1
            K   = svar*exp(-0.5*euclidean_distance(x,x,ls));    
            Ks = svar*exp(-0.5*euclidean_distance(x,y,ls));  
        end
    elseif strcmp(dist,'pearson')  
        if size(ls,1) == 1
            K   = svar*exp(-0.5*distance_pearson(x,x)/ls);
            Ks = svar*exp(-0.5*distance_pearson(x,y)/ls); 
        end

        if size(ls,1) > 1
            K   = svar*exp(-0.5*distance_pearson(x,x,ls));    
            Ks = svar*exp(-0.5*distance_pearson(x,y,ls));  
        end
    end
    
    Kss  = svar*ones(size(y,1),1);
   
end