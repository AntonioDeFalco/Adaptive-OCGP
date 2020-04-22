function distmat=distance_pearson(varargin)
    
    x=varargin{1};
    y=varargin{2};
    
    distmat = zeros( size(x,1), size(y,1) );
    
    if nargin==3
        ls=varargin{3};  
        for i=1:size(x,1)
            for j=1:size(y,1)
                R = corrcoef(x(i,:),y(j,:));
                buff = (1-R(1,2))/ls(i);
                distmat(i,j)= buff;
            end
        end
    else
        for i=1:size(x,1)
            for j=1:size(y,1)
                R = corrcoef(x(i,:),y(j,:));
                buff = (1-R(1,2));
                distmat(i,j)= buff;
            end
        end
    end

%{
function distmat=distance_pearsonold(x,y)
    distmat = zeros( size(x,1), size(y,1) );
    for i=1:size(x,1)
        for j=1:size(y,1)
            R = corrcoef(x(i,:),y(j,:));
            buff = (1-R(1,2));
            distmat(i,j)= buff;
        end
    end
end
%}