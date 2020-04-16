function distmat=euclidean_distance(x,y)
    distmat = zeros( size(x,1), size(y,1) );
    for i=1:size(x,1)
        for j=1:size(y,1)
            buff=(x(i,:)-y(j,:));   
            distmat(i,j)=buff*buff';
        end
    end
end