function distmat=euclidean_distance(varargin)
%
% Syntax:       distmat=euclidean_distance(varargin)
%               
% Inputs:       varargin: 
%                             
%               x sample
%              
%               y sample
% 
%               ls vector of lenght-scale hyperparameters (Optional for Adaptive Kernel) 
%
% Outputs:      distmat distance matrix
%                              
% Description:  Euclidean distance
%                      
% Author:       Antonio De Falco           
% 
    x=varargin{1};
    y=varargin{2};
    
    distmat = zeros( size(x,1), size(y,1) );
    
    if nargin==3
        ls=varargin{3};
        for i=1:size(x,1)
            for j=1:size(y,1)
                buff=(x(i,:)-y(j,:));
                buff=buff/ls(i);
                distmat(i,j)=buff*buff';
            end
        end
    else
        for i=1:size(x,1)
            for j=1:size(y,1)
                buff=(x(i,:)-y(j,:));   
                distmat(i,j)=buff*buff';
            end
        end
    end
end