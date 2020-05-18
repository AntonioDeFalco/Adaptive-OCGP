function [K,Ks,Kss]=scaled_exp_similarity_kernel(svar,x,y,dist_xn,dist_yn,mu)

    K   =  svar*exp(-0.5*euclidean_distance_similarity(x,x,mu,dist_xn,dist_xn));
    Ks = svar*exp(-0.5*euclidean_distance_similarity(x,y,mu,dist_xn,dist_yn));  
    Kss  = svar*ones(size(y,1),1);
    
end