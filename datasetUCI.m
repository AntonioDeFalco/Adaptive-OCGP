% Description:  testing on UCI datasets, comparing the proposed kernels and
% implementation of the hyperparameter selection of Xiao et al.
%
% Author:       Antonio De Falco           
% 

addpath('./SMRS_v1.0');
addpath('./GPR_OCC');

delete myData.xls

%% OPTIONS 

logtrasform = false;         %log transform of features with heavy-tailed distribution
scale = true;                %min-max normalization
norm_zscore = false;         %z-score normalization

sparse_selection = false;    %perform Sparse Features Selection

pca_exc = false;             %perform PCA 
perc_pca = 80;              %perform PCA

exec_SFS = false;           %perform Sequential forward selection (SFS) 
load_featuresSFS = false;   %load the features selected with SFS

%distance_mode = 'euclidean';%Use Euclidean distance    
distance_mode = 'pearson';   %Use Pearson distance (1- Pearson correlation coefficient) 

data_process = 'before';     %process data before computing sigma 
%data_process = 'after';     %process data after computing sigma 

%kernel = 'scaled';         %scaled exponential similarity kernel "Similarity network fusion for aggregating data types on a genomic scale"
kernel = 'adaptive';        %adaptive Gaussian kernel "MAGIC: A diffusion-based imputation method reveals gene-gene interactions in single-cell RNA-sequencing data"
%kernel = 'hyperOCC';       %Hyperparameter for SE kernel "Hyperparameter Selection for Gaussian Process One-Class Classification"

log_sigma = false;          %Sigma transform     

k_adapt = 15;               %k of Adaptive kernel

k_scaled = 30;              %k of Scaled kernel usually (10~30)
mu_scaled = 0.8;            %hyperparameter, usually (0.3~0.8)

%% Load Dataset 

tot_table = table();

data_folder = dir('./UCI_OCC_DATASETS/');

%for each UCI dataset 
for j = 4:12 
    
    AUC_mean = [];
    AUC_var = [];
    AUC_pred = [];
    AUC_ratio = [];
    
    name = ['./UCI_OCC_DATASETS/',data_folder(j).name]
    load(name);

    dataset = x;
    
    %iterations of dataset subdivision
    for i=1:20

        
        class1 = [];
        class2 = [];

        for i=1:size(dataset.nlab,1)
            if dataset.nlab(i) == 1
                class1 = [class1;dataset.data(i,:)];
            else
                class2 = [class2;dataset.data(i,:)];
            end
        end

        [~,I] = max([size(class1,1),size(class2,1)]);

        if I == 2
            temp = class1;
            class1 = class2;
            class2 = temp;
        end      

        trainRatio = 0.8;
        valRatio = 0.0;
        testRatio = 0.2;

        [trainInd,valInd,testInd] = dividerand(size(class1,1),trainRatio,valRatio,testRatio);

        x = class1(trainInd,:);
        t = [class1(testInd,:);class2];

        n = size(x,1);
        y = [ones(1,n)]';
        
        %k_adapt = floor((n/100)*2)

        t_label = [ones(1,size(testInd,2))';zeros(1,size(class2,1))'];

        %% Data processing

        %Sequential forward selection (SFS)
        if load_featuresSFS
            load('sel_features.mat');
            sel_features = sort(sel_features);
            x = x(:,sel_features);
            t = t(:,sel_features);
        end
        
        if strcmp(data_process,'before')
            [x,t] = data_processing(x,t,scale,norm_zscore,sparse_selection,pca_exc,perc_pca);
        end

        if strcmp(kernel,'adaptive')

            if strcmp(distance_mode,'euclidean')
                %ka = 30;
                [idx, dist] = knnsearch(x, x, 'k', k_adapt);%,'Distance','jaccard');
                if log_sigma
                    sigma = log(dist(:,k_adapt));
                else
                    sigma = dist(:,k_adapt);
                    %sigma = mean(dist(:,[2:k_adapt]),2);
                end     
            else %pearson distance
                dist=distance_pearson(x,x);
                dist = sort(dist,2);
                sigma = exp(dist(:,k_adapt));
            end

        end

        if strcmp(kernel,'hyperOCC')
            try
                sigma = hyperparameter_Selection(x);
            catch exception
                disp('continue')
                disp(i)
                continue
            end
            
            if log_sigma
                sigma = log(sigma);
            end
        end

        %Scaled Exponential Similarity Kernel
        if strcmp(kernel,'scaled')
            [~, dist] = knnsearch(x, x, 'k', k_scaled);
            dist_xn = mean(dist,2);
            [~, dist] = knnsearch(x, t, 'k', k_scaled);
            dist_yn = mean(dist,2);
        end


        if strcmp(data_process,'after')
            [x,t] = data_processing(x,t,scale,norm_zscore,sparse_selection,pca_exc,perc_pca);
        end

        if exec_SFS
            sel_features = SFS(x,t,t_label,sigma,distance_mode,score_mode);
            x = x(:,sel_features);
            t = t(:,sel_features);
        end

        %% Kernel

        %Signal Variance
        ins_pwr = x .^ 2;
        var_pwr = sum(ins_pwr)/length(x) - (sum(x) / length(x)).^2;
        svar = exp(2*log(var_pwr));
        svar = mean(svar);

        svar = 0.000045;
        
        if strcmp(kernel,'scaled')
            [K,Ks,Kss]=scaled_exp_similarity_kernel(svar,x,t,dist_xn,dist_yn,mu_scaled);
        else %hyperOCC or adaptive
            if strcmp(distance_mode,'euclidean')
                [K,Ks,Kss]=se_kernel_adaptive(svar,sigma,x,t,'euclidean');
            else %pearson distance
                [K,Ks,Kss]=se_kernel_adaptive(svar,sigma,x,t,'pearson');
            end
        end

        %% GP OCC
        modes={'mean','var','pred','ratio'};
        titles={'mean \mu_*','neg. variance -\sigma^2_*','log. predictive probability p(y=1|X,y,x_*)','log. moment ratio \mu_*/\sigma_*'};

        min_scores  = [];
        max_scores  = [];
        scores = [];
        AUCs = [];

        try
        for i=1:4
            %compute scores        
            score=GPR_OCC(K,Ks,Kss,modes{i});
             
            [X,Y,~,AUC] = perfcurve(t_label,score,1);

            AUCs = [AUCs,AUC]; 
        end

        catch exception
            disp('continue')
            continue
        end
        
        AUC_mean = [AUC_mean;AUCs(1)];
        AUC_var = [AUC_var;AUCs(2)];
        AUC_pred = [AUC_pred;AUCs(3)];
        AUC_ratio = [AUC_ratio;AUCs(4)];

    end

    mean(AUC_mean)
    mean(AUC_var)
    mean(AUC_pred)
    mean(AUC_ratio)
   
    tab = table(string(data_folder(j).name),mean(AUC_mean),mean(AUC_var),mean(AUC_pred),mean(AUC_ratio),size(AUC_mean,1));

    tot_table =  [tot_table;tab];

end    

writetable(tot_table,'myData.xls');