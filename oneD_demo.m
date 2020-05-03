addpath('./GPR_OCC');

x_obs = [-8, -1.5, -1.2, -1, 0, 1.5, 2.5, 2.7, 3, 8]';

y_obs = ones(1,size(x_obs,1))';

x_s = linspace(-8, 8, 80)';

%[x_obs,x_s] = data_processing(x_obs,x_s,false,true,false,false,0);

svar = 0.3;

ka = 2;
[idx, dist] = knnsearch(x_obs, x_obs, 'k', ka);%,'Distance','jaccard');
sigma_ada = dist(:,ka);

%sigma = hyperparameter_Selection(x_obs);

sigma_arr = [1, 0.3];
titles = {'Adaptive', 'l=1', 'l=0.3'};

for i=1:3
    
    if i == 1
        sigma = sigma_ada;
    else
        sigma = sigma_arr(i-1);
    end

    text = titles(i);

    [K,Ks,Kss]=se_kernel(svar,sigma,x_obs, x_s,'euclidean');
    sigma=GPR_OCC(K,Ks,Kss,'var');
    mu=GPR_OCC(K,Ks,Kss,'mean');

    K_sTKinv = transpose(Ks)*pinv(K);
    mu_s = K_sTKinv*y_obs;
    Sigma_s = Kss - K_sTKinv*Ks;

    figure()
    hold on;

    stds = sqrt(diag(Sigma_s));

    err_xs = [x_s; flip(x_s, 1)];

    err_ys = [mu_s + 2 * stds; flip(mu_s - 2 * stds, 1)];

    patch(err_xs, err_ys, 'yellow');

    line(x_s, mu_s);
    plot(x_obs, y_obs,'r*')
    title(text)
    xlabel('x_*') 
    ylabel('f_*') 
    legend('training samples','variance','mean')
end
