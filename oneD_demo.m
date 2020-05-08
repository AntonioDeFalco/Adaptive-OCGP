close all

addpath('./GPR_OCC');

x_obs = [-7, -2.5, -2, -1.5, -1, 1, 1.2, 1.3,1.4, 1.5,1.6,1.7,1.8, 1.9,2.0, 2.7,3.0]'; %OUTLIER

%x_obs = [-2.5, -2, -1.5, -1, 1, 1.2, 1.3,1.4, 1.5,1.6,1.7,1.8, 1.9,2.0, 2.7,3.0]'; %WITHOUT OUTLIER

%x_obs = [-6,-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,6]'; %DISTANZA 1

y_obs = ones(1,size(x_obs,1))';

x_s = linspace(-8, 8, 80)';

svar = 1.0; %0.3

%adaptive
ka = 2;
[idx, dist] = knnsearch(x_obs, x_obs, 'k', ka);
sigma_ada = dist(:,ka);

%scalaed
k = 30;     %number of neighbors, usually (10~30)
mu = 0.6;   %hyperparameter, usually (0.3~0.8)
[~, dist] = knnsearch(x_obs, x_obs, 'k', k);
dist_xn = mean(dist,2);
[~, dist] = knnsearch(x_obs, x_s, 'k', k);
dist_yn = mean(dist,2);

%Xiao
sigma_xiao = hyperparameter_Selection(x_obs);

sigma_arr = [sigma_xiao, 1, 0.3];
titles = {'Adaptive ','XiaoSelection ', 'l=1 ', 'l=0.3 ','Scaled '};

for i=1:4
    
    if i == 1
        sigma = sigma_ada;
    elseif i<5
        sigma = sigma_arr(i-1);
    end
    
    text = titles(i);

    if i == 5
    [K,Ks,Kss]=scaled_exp_similarity_kernel(svar,x_obs,x_s,dist_xn,dist_yn,mu);
    else
    [K,Ks,Kss]=se_kernel(svar,sigma,x_obs, x_s,'euclidean');
    end
        
    noise=0.01;
    K=K+noise*eye(size(K));
    Kss=Kss+noise*ones(size(Kss));
    
    L = chol(K)';   
    alpha = L'\(L\ones(size(K,1),1));
    mu_s = Ks' * alpha;  
    
    cov_s = Kss - Ks'*inv(K)*Ks;
    stds = diag(cov_s);
    
    figure()
    hold on;
   
    err_xs = [x_s; flip(x_s, 1)];

    err_ys = [mu_s + 2 * stds; flip(mu_s - 2 * stds, 1)];

    patch(err_xs, err_ys, 'yellow');

    line(x_s, mu_s);
    plot(x_obs, y_obs,'r*')
    title([char(text),'SignalVar ',sprintf('%0.2f',svar),' Noise ',sprintf('%0.2f',noise)])
    xlabel('x_*') 
    ylabel('f_*') 
    legend('variance','mean','training samples')
end