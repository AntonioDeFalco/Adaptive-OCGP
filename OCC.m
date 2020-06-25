addpath('./gpocc');
addpath('./SMRS_v1.0');

%% OPTIONS 

logtrasform = true;         %log transform features with heavy-tailed distribution
scale = true;               %min-max normalization
norm_zscore = false;        %z-score normalization

pca_exc = true;             %perform PCA 
perc_pca = 80;              %variance percentage

sparse_selection = false;   %perform Sparse Features Selection

exec_SFS = false;           %perform Sequential forward selection (SFS) 
exec_SBS = false;           %perform Sequential Backward Selection (SBS) 
score_mode = 'mean';        %sequential selection criterion (mean, var)
load_featuresSFS = true;    %load the features selected with SFS

distance_mode = 'euclidean'; %Use Euclidean distance    
%distance_mode = 'pearson';  %Use Pearson distance (1- Pearson correlation coefficient) 

%data_process = 'before';   %preprocessing data before computing sigma 
data_process = 'after';     %preprocessing data after computing sigma 

%kernel = 'scaled';         %Scaled exponential similarity kernel
kernel = 'adaptive';        %Adaptive Gaussian kernel
%kernel = 'hyperOCC';       %Hyperparameter Selection of Xiao et al.

log_sigma = true;           %Log-trasform Sigma     
k_adapt = 30;               %k of Adaptive kernel

k_scaled = 30;              %k of Scaled kernel usually (10~30)
mu_scaled = 0.8;            %hyperparameter, usually (0.3~0.8)

%% Load Dataset 

%Training Set
if not(exist('T'))
    load('T.mat')
end

%All features
if not(exist('A'))
    load('A.mat')
end

%Validation Set
if not(exist('V'))
    load('V.mat')
end

%Continuous features
features = [2:42,44:46,48,52:71];

%Features to log transform
if logtrasform
    features_log = [2,3,7:10,41,42,44,45,46,48,58];
else
    features_log = [];
end

x=read_features(T,features,features_log);
t=read_features(A,features,features_log);

%ONE-HOT ENCODING
x_categ = add_categorical_onehot(T);
x = [x,x_categ];
t_categ = add_categorical_onehot(A);
t = [t,t_categ];

%FREQUENCY ENCODING
x_categ = add_categorical_frequency(T);
x = [x,x_categ];
t_categ = add_categorical_frequency(A);
t = [t,t_categ];

%LABEL
n=size(x,1);
y = [ones(1,n)]';
[~,ia,~] = intersect(A(:,1), V(:,1));
n=size(t,1);
t_label = zeros(1,n)';
for i=ia
    t_label(i) = 1;
end

%% Data pre-processing

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
        [idx, dist] = knnsearch(x, x, 'k', k_adapt);
        if log_sigma
            sigma = log(dist(:,k_adapt));
        else
            sigma = dist(:,k_adapt);
        end     
    else %pearson distance
        dist=distance_pearson(x,x);
        dist = sort(dist,2);
        %sigma = dist(:,k_adapt);
        sigma = exp(dist(:,k_adapt));
    end
    
end

if strcmp(kernel,'hyperOCC')
    sigma = hyperparameter_Selection(x);
end

%Scaled Exponential Similarity Kernel
if strcmp(kernel,'scaled')
    [~, dist] = knnsearch(x, x, 'k', k_scaled);
    dist_xn = mean(dist,2);
    [~, dist] = knnsearch(x, t, 'k', k_scaled);
    dist_yn = mean(dist,2);
end

if exec_SFS
    sel_features = SFS(x,t,t_label,sigma,distance_mode,score_mode);
    x = x(:,sel_features);
    t = t(:,sel_features);
end

if exec_SBS
    sel_features = SBS(x,t,t_label,sigma,distance_mode,score_mode);
    x = x(:,sel_features);
    t = t(:,sel_features);
end

if strcmp(data_process,'after')
    [x,t] = data_processing(x,t,scale,norm_zscore,sparse_selection,pca_exc,perc_pca);
end


%% Kernel

%Signal Variance
ins_pwr = x .^ 2;
var_pwr = sum(ins_pwr)/length(x) - (sum(x) / length(x)).^2;
svar = exp(2*log(var_pwr));
svar = mean(svar);

svar = 0.0045;

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
for i=1:4
    %compute scores
    score=GPR_OCC(K,Ks,Kss,modes{i});
    [X,Y,~,AUC] = perfcurve(t_label,score,1);
    figure(i)
    plot(X,Y)
    xlabel('False positive rate') 
    ylabel('True positive rate')
    title(sprintf('ROC %s',titles{i}))
    text(0.75,0.1,sprintf('AUC=%0.4f',AUC),'FontSize',14);
    
    min_score = min(score);
    max_score = max(score);
    min_scores = [min_scores,min_score];
    max_scores = [max_scores,max_score];
    scores = [scores,score];
    AUCs = [AUCs,AUC]; 
end

AUC_mean = AUCs(1);
AUC_var = AUCs(2);
AUC_pred = AUCs(3);
AUC_ratio = AUCs(4);

min_mean = min_scores(:,1);
min_var = min_scores(:,2);
min_pred = min_scores(:,3);
min_ratio = min_scores(:,4);

max_mean = max_scores(:,1);
max_var = max_scores(:,2);
max_pred = max_scores(:,3);
max_ratio = max_scores(:,4);

t_score = table(AUC_mean,AUC_var,AUC_pred,AUC_ratio,min_mean,min_var,min_pred,min_ratio,max_mean,max_var,max_pred,max_ratio);

writetable(t_score,'myData.xls');




