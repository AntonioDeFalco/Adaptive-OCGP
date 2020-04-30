function sel_features=SFS(x,t,t_label,sigm,distance_mode,score_mode)

x_old = x;
t_old = t;

sel_features = [];
all_features = [1:size(x,2)];
last_AUC = 0;

    for i=1:size(x,2)

        AUCC = [];

            for j = all_features

            x = x_old(:,[sel_features,j]);
            t = t_old(:,[sel_features,j]);

            ins_pwr = x .^ 2;
            var_pwr = sum(ins_pwr)/length(x) - (sum(x) / length(x)).^2;
            svar = exp(2*log(var_pwr));
            svar = mean(svar);
            
            svar = 0.0045;

            [K,Ks,Kss]=se_kernel(svar,sigm,x,t,distance_mode);

            score=GPR_OCC(K,Ks,Kss,score_mode);
            [X,Y,~,AUC] = perfcurve(t_label,score,1);

            AUCC = [AUCC,AUC];

            end

        [max_auc,max_ind] = max(AUCC);

        if last_AUC > max_auc
            break
        end

        last_AUC = max_auc
        sel_features = [sel_features,all_features(max_ind)];
        all_features(max_ind) = [];   

    end
end