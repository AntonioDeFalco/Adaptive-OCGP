function [K,Ks,Kss]=scaled_exp_similarity_kernel(svar,x,y,dist_xn,dist_yn,mu)
    %ls   = exp(2*loghypers(1));
    %svar = exp(2*loghypers(2));
    
    %ls = loghypers;
    
    %svar = 0.0498;
    
    %K   = svar*exp(-0.5*euclidean_distance_similarity(x,x,ls));
    %Ks = svar*exp(-0.5*euclidean_distance_similarity(x,y,ls));  
    
    %{
    k = 60;%number of neighbors, usually (10~30)
    mu = 0.6; %hyperparameter, usually (0.3~0.8)
    
    [idx, dist] = knnsearch(x, x, 'k', k);
    dist_xn = mean(dist,2);
    
    %mu = dist(:,k);
    
    [idx, dist] = knnsearch(x, y, 'k', k);
    dist_yn = mean(dist,2);
    %}
    
    %svar = 0.001;
    K   =  svar*exp(-0.5*euclidean_distance_similarity(x,x,mu,dist_xn,dist_xn));
    Ks = svar*exp(-0.5*euclidean_distance_similarity(x,y,mu,dist_xn,dist_yn));  
   
    
    Kss  = svar*ones(size(y,1),1);
end