function [K,Ks,Kss]=se_kernel(svar,ls,x,y,dist)
%
% Syntax:       [K,Ks,Kss]=se_kernel(svar,ls,x,y,dist)
%               
% Inputs:       svar signal variance hyperparameter is an (n x d) matrix of traning set containing n samples of d-dimensional
%              
%               ls lenght-scale hyperparameter (single value or vector for Adaptive Kernel)
%               
%               x is an (n x d) matrix of traning set containing n samples of d-dimensional
%              
%               t is an (n x d) matrix of test set containing n samples of d-dimensional 
% 
%               dist distance to be used in the kernel
%                
% Outputs:      [K,Ks,Kss] covariance matrices
%                              
% Description:  Kernel Squared Exponential computation or Adaptive Kernel if lenght-scale is not single value
%                      
% Author:       Antonio De Falco           
%            

    if strcmp(dist,'euclidean')
        if size(ls,1) == 1
            K   = svar*exp(-0.5*euclidean_distance(x,x)/ls);
            Ks = svar*exp(-0.5*euclidean_distance(x,y)/ls); 
        end
        %Adaptive Kernel
        if size(ls,1) > 1      
            K = svar*exp(-0.5*euclidean_distance(x,x,ls));           
            K = (K + K')/2; %Symmetrization
            Ks = svar*exp(-0.5*euclidean_distance(x,y,ls));
        end
    elseif strcmp(dist,'pearson')  
        if size(ls,1) == 1
            K   = svar*exp(-0.5*distance_pearson(x,x)/ls);
            Ks = svar*exp(-0.5*distance_pearson(x,y)/ls); 
        end
        %Adaptive Kernel
        if size(ls,1) > 1
            K   = svar*exp(-0.5*distance_pearson(x,x,ls));
            K = (K + K')/2; %Symmetrization
            Ks = svar*exp(-0.5*distance_pearson(x,y,ls));  
        end
    end
    
    Kss  = svar*ones(size(y,1),1);
   
end