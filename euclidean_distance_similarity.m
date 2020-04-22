function distmat=euclidean_distance_similarity(x,y,mu,dist_xn,dist_yn)
    distmat = zeros( size(x,1), size(y,1) );
    for i=1:size(x,1)
        for j=1:size(y,1)
            dist=sqrt((x(i,:)-y(j,:)));
            dist2=dist.^2;
            epsilon_ij = (dist_xn(i) + dist_yn(j) + dist)/3;
            buff=dist2./(mu*epsilon_ij);
            distmat(i,j)=buff*buff';
        end
    end
end