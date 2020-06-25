function distmat=euclidean_distance(varargin)

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