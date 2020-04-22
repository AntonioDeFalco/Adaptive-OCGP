function sigma=hyperparameter_Selection(x)

    n=size(x,1);
    k=ceil(5 * log(n));
    [ind_x_ij,D]=knnsearch(x,x,'K',k);
    ind_x_ij(:,1) = [];
    [ll,D]=knnsearch(x,x,'K',n);
    D(:,1) = [];

    d_min = min(D(:));
    d_max = max(D(:));

    logd_min = log(d_min);
    %logd_min = log(0);
    logd_max = log(d_max);

    %Candidate set
    L = exp(linspace(logd_min,logd_max,20));

    x_ij = cell(n,k-1);

    for i=1:n
        for j=1:k-1  
                val = x(ind_x_ij(i,j),:);
                x_ij{i,j} = val;
        end
    end

    vn_i=[];

    for i=1:n
        vu_ij=[];
        for j=1:k-1  
            val = (x_ij{i,j} - x(i,:))/(norm(x_ij{i,j} - x(i,:)));
            %val = (x(x_ij(i,j),:) - x(i,:))/(norm(x(x_ij(i,j),:)) - x(i,:));
            vu_ij = [vu_ij;val];
        end
            vn_i=[vn_i;sum(vu_ij)];
     end

    %vu(i,j)=(x(:,x_ij(i,j)) - x(:,i))/(norm(x(:,x_ij(i,j)) - x(:,i)))
    teta_ij=[];

    for i=1:n
        teta_j = [];
        for j=1:k-1 
                val = dot((x_ij{i,j} - x(i,:))',vn_i(i,:));
                %val = (x(x_ij(i,j),:) - x(i,:))'*vn_i(i,:);
                teta_j = [teta_j,val];
        end
        teta_ij = [teta_ij;teta_j];
    end

    eta_i =[];

    for i=1:n
            val = 1/k * length(nonzeros(teta_ij(teta_ij(i,:)>0)));
            eta_i = [eta_i;val];
    end

    %x2 = -ones(size(x1,1),size(x1,2));

    gamma=(n/100)*5;
    m = ceil(gamma);

    [eta_Sort,ind_sort] = sort(eta_i) ;

    interior_samples = x(ind_sort(1:m),:);
    edge_samples = x(ind_sort(n-(m-1):n),:);


    y = [ones(1,n)]';
    meanfunc = @meanConst; hyp.mean = 0;
    covfunc = @covSEisoU; 
    likfunc = @likErf;

    KL = [];
    
    ins_pwr = x .^ 2;
    var_pwr = sum(ins_pwr)/length(x) - (sum(x) / length(x)).^2;
    svar = exp(2*log(var_pwr));
    svar = mean(svar);

    for j=1:size(L,2)  
        %hyp.cov = L(j);
        %[ymu ys2 mu_inter sigma_inter lp] = gp(hyp, @infLaplace, meanfunc, covfunc, likfunc, x, y, interior_samples, ones(n,1));
        %[ymu ys2 mu_edge sigma_edge lp] = gp(hyp, @infLaplace, meanfunc, covfunc, likfunc, x, y, edge_samples, zeros(n,1));

        [K,Ks,Kss]=se_kernel(svar,L(j),interior_samples, ones(n,1));
        sigma_inter=GPR_OCC(K,Ks,Kss,'var');
        mu_inter=GPR_OCC(K,Ks,Kss,'mean');
        [K,Ks,Kss]=se_kernel(svar,L(j),edge_samples, zeros(n,1));
        sigma_edge=GPR_OCC(K,Ks,Kss,'var');
        mu_edge=GPR_OCC(K,Ks,Kss,'mean');
        
        summ = 0;
            for j=1:m  
                    summ = summ + (2*log(sigma_edge(j)) -  2*log(sigma_inter(j))*(sigma_inter(j)/sigma_edge(j))+((mu_inter(j)-mu_edge(j))/sigma_edge(j))^2-1);
            end
        val = 1/2 * summ;
        KL = [KL;val];
    end

    [max_kl,ind_max_kl] = max(KL);
    sigma = L(ind_max_kl);

end