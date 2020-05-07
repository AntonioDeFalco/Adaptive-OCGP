close all

addpath('./GPR_OCC');

%x_obs = [-8, -1.5, -1.2, -1, 0, 1.5, 2.5, 2.7,3,8]'; %PRIMO

x_obs = [-7, -2.5, -2, -1.5, -1, 1, 1.2, 1.3,1.4, 1.5,1.6,1.7,1.8, 1.9,2.0, 2.7,3.0]'; %SECONDO

%x_obs = [-6,-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,6]'; %DISTANZA 1

%x_obs = [-6, -4, -2, -1, 1,1.5, 2, 2.5, 3,3.5, 4,4.5, 5,5.5, 6]';

y_obs = ones(1,size(x_obs,1))';

x_s = linspace(-8, 8, 80)';

[x_obs,x_s] = data_processing(x_obs,x_s,false,true,false,false,0);

%svar = 1.0;
svar = 0.3;

ka = 2;
[idx, dist] = knnsearch(x_obs, x_obs, 'k', ka);
sigma_ada = dist(:,ka);

sigma_xiao = hyperparameter_Selection(x_obs);

sigma_arr = [sigma_xiao, 1, 0.3];
titles = {'Adaptive','XiaoSelection', 'l=1', 'l=0.3'};

for i=1:4
    
    if i == 1
        sigma = sigma_ada;
    else
        sigma = sigma_arr(i-1);
    end

    text = titles(i);

    [K,Ks,Kss]=se_kernel(svar,sigma,x_obs, x_s,'euclidean');
    
    noise=0.01;
    K=K+noise*eye(size(K));
    Kss=Kss+noise*ones(size(Kss));
    
    L = chol(K)';   
    alpha = L'\(L\ones(size(K,1),1));
    mu_s = Ks' * alpha;  
    v = L\Ks;
    stds = -Kss + sum(v .* v)'; 
    
    %mu_s2=GPR_OCC(K,Ks,Kss,'mean');
    %Sigma_s2=GPR_OCC(K,Ks,Kss,'var');
    
    %K_sTKinv = transpose(Ks)*pinv(K);
    %mu_s = K_sTKinv*y_obs;
    %Sigma_s = Kss - K_sTKinv*Ks;
    %stds = sqrt(diag(Sigma_s));
    
    figure()
    hold on;
   
    err_xs = [x_s; flip(x_s, 1)];

    err_ys = [mu_s + 2 * stds; flip(mu_s - 2 * stds, 1)];

    patch(err_xs, err_ys, 'yellow');

    line(x_s, mu_s);
    plot(x_obs, y_obs,'r*')
    title(text)
    xlabel('x_*') 
    ylabel('f_*') 
    legend('variance','mean','training samples')
end
