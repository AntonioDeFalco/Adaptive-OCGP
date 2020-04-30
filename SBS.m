function sel_features=SBS(x,t,t_label,sigm,distance_mode,score_mode)

x_old = x;
t_old = t;


all_features = [1:size(x,2)];
sel_features = all_features;

ins_pwr = x .^ 2;
var_pwr = sum(ins_pwr)/length(x) - (sum(x) / length(x)).^2;
svar = exp(2*log(var_pwr));
svar = mean(svar);

svar = 0.0045;

[K,Ks,Kss]=se_kernel(svar,sigm,x,t,distance_mode);

score=GPR_OCC(K,Ks,Kss,score_mode);
[~,~,~,AUC] = perfcurve(t_label,score,1);

last_AUC = AUC

    for i=1:size(x,2)

        AUCC = [];
        
        found = false;

            for j = all_features

            sel_features = all_features;
            sel_features(sel_features == j) = [];
            size(sel_features,2)

            x = x_old(:,sel_features);
            t = t_old(:,sel_features);

            ins_pwr = x .^ 2;
            var_pwr = sum(ins_pwr)/length(x) - (sum(x) / length(x)).^2;
            svar = exp(2*log(var_pwr));
            svar = mean(svar);
            
            svar = 0.0045;

            [K,Ks,Kss]=se_kernel(svar,sigm,x,t,distance_mode);

            score=GPR_OCC(K,Ks,Kss,score_mode);
            [~,~,~,AUC] = perfcurve(t_label,score,1);

            AUCC = [AUCC,AUC];
            
            AUC
            
            if last_AUC < AUC
                 last_AUC = AUC;
                 found = true;
                 disp("last_AUC >= AUC")
                 all_features(all_features == j) = [];
                 break
            end

            end
        
        if found == false
            break
        end
        
        %{    
        [max_auc,max_ind] = max(AUCC);

        if last_AUC < max_auc
            break
        end

        last_AUC = max_auc
        all_features(max_ind) = [];
        %}

    end
end