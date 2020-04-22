function [K,Ks,Kss]=se_kernel(svar,ls,x,y)

    if size(ls,1) == 1
        K   = svar*exp(-0.5*euclidean_distance(x,x)/ls);
        Ks = svar*exp(-0.5*euclidean_distance(x,y)/ls); 
    end
    
    if size(ls,1) > 1
        K   = svar*exp(-0.5*euclidean_distance_danapeer(x,x,ls));    
        Ks = svar*exp(-0.5*euclidean_distance_danapeer(x,y,ls));  
    end
    
    Kss  = svar*ones(size(y,1),1);
   
end