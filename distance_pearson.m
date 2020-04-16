function distmat=distance_pearson(x,y)
    distmat = zeros( size(x,1), size(y,1) );
    for i=1:size(x,1)
        for j=1:size(y,1)
            R = corrcoef(x(i,:),y(j,:));
            buff = (1-R(1,2));
            distmat(i,j)= buff;
        end
    end
end