
logtrasform = true;
scale = true;
norm_zscore = false;

pca_exc = false;
perc_pca = 80;

exec_SFS = false;
load_featuresSFS = false;

%distance_mode = 'euclidean';
distance_mode = 'pearson';

%data_process = 'before';
data_process = 'after';

%kernel = 'scaled';   %scaled exponential similarity kernel "Similarity network fusion for aggregating data types on a genomic scale"
kernel = 'adaptive'; %adaptive Gaussian kernel "MAGIC: A diffusion-based imputation method reveals gene-gene interactions in single-cell RNA-sequencing data"
%kernel = 'hyperOCC'; %Hyperparameter for SE kernel "Hyperparameter Selection for Gaussian Process One-Class Classification"


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

features = [2:42,44:46,48,52:71];

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

%Sequential forward selection (SFS)
if load_featuresSFS
    load('sel_features.mat');
    sel_features = sort(sel_features);
    x = x(:,sel_features);
    t = t(:,sel_features);
end


%{
load('/Users/anthony/Dropbox/tesi/gpml-matlab-master/doc/sel_features_sparse.mat');
sel_features = repInd;
x = x(:,sel_features);
t = t(:,sel_features);
%}

if strcmp(data_process,'before')
    [x,t] = data_processing(x,t,scale,norm_zscore,pca_exc,perc_pca);
end

if strcmp(kernel,'adaptive')
    
    if strcmp(distance_mode,'euclidean')
        ka = 30;
        [idx, dist] = knnsearch(x, x, 'k', ka);%,'Distance','jaccard');
        sigma = dist(:,ka);
        %sigma = log(dist(:,ka));
    else %pearson distance
        dist=distance_pearson(x,x);
        dist = sort(dist,2);
        sigma = exp(dist(:,ka));
    end
    
end

if strcmp(kernel,'hyperOCC')
    sigma = hyperparameter_Selection(x);
end

%Scaled Exponential Similarity Kernel
if strcmp(kernel,'scaled')
    k = 30;%number of neighbors, usually (10~30)
    mu = 0.6; %hyperparameter, usually (0.3~0.8)
    [~, dist] = knnsearch(x, x, 'k', k);
    dist_xn = mean(dist,2);
    [~, dist] = knnsearch(x, t, 'k', k);
    dist_yn = mean(dist,2);
end

%{
dist=distance_pearson(x,x);
dist = sort(dist,2);
sigma = exp(dist(:,ka));
%}

if strcmp(data_process,'after')
    [x,t] = data_processing(x,t,scale,norm_zscore,pca_exc,perc_pca);
end

if exec_SFS
    sel_features = SFS(x,t,t_label,sigma);
    x = x(:,sel_features);
    t = t(:,sel_features);
end

modes={'mean','var','pred','ratio'};
titles={'mean \mu_*','neg. variance -\sigma^2_*','log. predictive probability p(y=1|X,y,x_*)','log. moment ratio \mu_*/\sigma_*'};

%Signal Variance
ins_pwr = x .^ 2;
var_pwr = sum(ins_pwr)/length(x) - (sum(x) / length(x)).^2;
svar = exp(2*log(var_pwr));
svar = mean(svar);

if strcmp(kernel,'scaled')
    [K,Ks,Kss]=scaled_exp_similarity_kernel(svar,x,t,dist_xn,dist_yn,mu);
else %hyperOCC or adaptive
    if strcmp(distance_mode,'euclidean')
        [K,Ks,Kss]=se_kernel(svar,sigma,x,t,'euclidean');
    else %pearson distance
        [K,Ks,Kss]=se_kernel(svar,sigma,x,t,'pearson');
    end
end

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

%mean = scores(:,1);
%var = scores(:,2);
%pred = scores(:,3);
%ratio = scores(:,4);

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




