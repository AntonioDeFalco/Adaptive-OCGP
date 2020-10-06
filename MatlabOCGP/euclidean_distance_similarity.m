function distmat=euclidean_distance_similarity(x,y,mu,dist_xn,dist_yn)
%
% Syntax:       distmat=euclidean_distance_similarity(x,y,mu,dist_xn,dist_yn)
%               
% Inputs:       svar signal variance hyperparameter is an (n x d) matrix of traning set containing n samples of d-dimensional
%                             
%               x sample
%              
%               y sample
% 
%               dist_xn distance matrix of sample x
%
%               dist_yn distance matrix of sample y
%
%               mu hyperparameter of Scaled Kernel
%
% Outputs:      distmat distance matrix
%                              
% Description:  Scaled Kernel Squared Exponential computation
%                      
% Author:       Antonio De Falco           
% 

    distmat = zeros( size(x,1), size(y,1) );
    for i=1:size(x,1)
        for j=1:size(y,1)
            dist=(x(i,:)-y(j,:));
            dist2 = dist*dist';
            dist = sqrt(dist2);
            epsilon_ij = (dist_xn(i) + dist_yn(j) + dist)/3;
            buff=dist2/(mu*epsilon_ij);
            distmat(i,j)=buff;
        end
    end
end