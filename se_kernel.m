function [K,Ks,Kss]=se_kernel(svar,ls,x,y)

    K   = svar*exp(-0.5*euclidean_distance_danapeer(x,x,ls));
    
    Ks = svar*exp(-0.5*euclidean_distance_danapeer(x,y,ls));  

    Kss  = svar*ones(size(y,1),1);
   
end