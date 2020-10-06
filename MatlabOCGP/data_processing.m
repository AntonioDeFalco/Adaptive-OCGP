function [x_new,y_new]=data_processing(x,y,scale,norm_zscore,sparse_selection,pca_exc,perc_pca)
%
% Syntax:       [x_new,y_new]=data_processing(x,y,scale,norm_zscore,sparse_selection,pca_exc,perc_pca)
%               
% Inputs:       x is an (n x d) matrix of traning set containing n samples of d-dimensional
%              
%               t is an (n x d) matrix of test set containing n samples of d-dimensional 
% 
%               scale boolean value perform min-max normalization
%
%               norm_zscore boolean value perform z-score normalization
%
%               sparse_selection boolean value perform Sparse Features Selection
%
%               pca_exc boolean value perform PCA
%
%               perc_pca variance percentage
%
% Outputs:      x is an (n x d) matrix of data processed of traning set
%              
%               t is an (n x d) matrix of data processed test set
%                              
% Description:  Several dataset preprocessing methods the method returns the
%               processed dataset data
%                      
% Author:       Antonio De Falco           
% 

    % Normalize z-score

    %
    %if norm_zscore
    %    all = normalize(all,2);  
    %end
    
    if norm_zscore

            [Ztrain,tr_mu,tr_sigma] = zscore(x); % Standardize the training data
            tr_sigma(tr_sigma==0) = 1;

            Ztest = (y-tr_mu)./tr_sigma; 

            x = Ztrain;
            y = Ztest;
    end

    %
    x_size = size(x,1);
    y_size = size(y,1);
    all = [x;y];

    if scale
        colmin = min(all);
        colmax = max(all);
        all = rescale(all,'InputMin',colmin,'InputMax',colmax);
    end

          
    %
    if sparse_selection

        % input the regularization parameter
        alpha = 20; % typically alpha in [2,20]

        % if desired to reduce data dimension by PCA enter the projection
        % dimension r, else r = 0 for using the data without any projections
        r = 0;%4;

        % report information about iterations
        verbose = true;

        % find the representatives via sparse modelling
        [repInd,~] = smrs(all,alpha,r,verbose);

        sel_features = repInd;

        all = all(:,sel_features);
    
    end

    % PCA

    if pca_exc
        [~,scoreTrain,~,~,explained,~] = pca(all);

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

    %

    x_new = all(1:x_size,:);
    y_new = all(x_size+1:y_size+x_size,:);
    
end