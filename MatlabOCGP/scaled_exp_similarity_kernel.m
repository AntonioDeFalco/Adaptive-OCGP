function [K,Ks,Kss]=scaled_exp_similarity_kernel(svar,x,y,dist_xn,dist_yn,mu)

%
% Syntax:       scaled_exp_similarity_kernel(svar,x,y,dist_xn,dist_yn,mu)
%               
% Inputs:       svar signal variance hyperparameter is an (n x d) matrix of traning set containing n samples of d-dimensional
%                             
%               x is an (n x d) matrix of traning set containing n samples of d-dimensional
%              
%               t is an (n x d) matrix of test set containing n samples of d-dimensional 
% 
%               dist_xn training set distance matrix
%
%               dist_yn test set distance matrix
%
%               mu hyperparameter of Scaled Kernel
%
% Outputs:      [K,Ks,Kss] covariance matrices
%                              
% Description:  Scaled Kernel Squared Exponential computation
%                      
% Author:       Antonio De Falco           
%            

    K   =  svar*exp(-0.5*euclidean_distance_similarity(x,x,mu,dist_xn,dist_xn));
    Ks = svar*exp(-0.5*euclidean_distance_similarity(x,y,mu,dist_xn,dist_yn));  
    Kss  = svar*ones(size(y,1),1);
    
end