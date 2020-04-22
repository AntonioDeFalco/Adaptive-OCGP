function [x_new,y_new]=data_processing(x,y,scale,norm_zscore,pca_exc,perc_pca)

    x_size = size(x,1);
    y_size = size(y,1);
    all = [x;y];

    if scale
        colmin = min(all);
        colmax = max(all);
        all = rescale(all,'InputMin',colmin,'InputMax',colmax);
    end

    %% Normalize z-score

    %
    if norm_zscore
        all = normalize(all,2);  
    end
    %% PCA

    if pca_exc
        [coeff,scoreTrain,~,~,explained,mu] = pca(all);

        if perc_pca
            sum_explained = 0;
            idx = 0;
            while sum_explained < perc_pca
                idx = idx + 1;
                sum_explained = sum_explained + explained(idx);
            end
        else
            idx = num_pca;
        end
        all = scoreTrain(:,1:idx);
    end

    %%

    x_new = all(1:x_size,:);
    y_new = all(x_size+1:y_size+x_size,:);
    
end