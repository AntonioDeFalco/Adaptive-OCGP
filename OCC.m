logtrasform = true;
scale = true;
norm_zscore = false;
pca_exc = true;
perc_pca = 80;
num_pca = 0;
exec_SFS = false;
load_featuresSFS = false;
hyperSelection = true;

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

k = 30;
ka = 30;
%colmin = min(x);
%colmax = max(x);
%xx = rescale(x,'InputMin',colmin,'InputMax',colmax);

[idx, dist] = knnsearch(x, x, 'k', k);%,'Distance','jaccard');
sigma = dist(:,ka);
%sigma = log(dist(:,ka));
%sigma = dist(:,ka);
%sigma = rescale(sigma)+0.01;

if hyperSelection
    sigma = hyperparameter_Selection(x);
end

%{
dist=distance_pearson(x,x);
dist = sort(dist,2);
sigma = exp(dist(:,ka));
%}

%% Scaled min max 
all = [x;t];

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

x = all(1:102,:);
t = all(103:20402,:);

if exec_SFS
    sel_features = SFS(x,t,t_label,sigma);
    x = x(:,sel_features);
    t = t(:,sel_features);
end

%{
dist=distance_pearson(x,x);
dist = sort(dist,2);
sigma = exp(dist(:,ka));
%}

modes={'mean','var','pred','ratio'};
titles={'mean \mu_*','neg. variance -\sigma^2_*','log. predictive probability p(y=1|X,y,x_*)','log. moment ratio \mu_*/\sigma_*'};

ins_pwr = x .^ 2;
var_pwr = sum(ins_pwr)/length(x) - (sum(x) / length(x)).^2;
svar = exp(2*log(var_pwr));
svar = mean(svar);

[K,Ks,Kss]=se_kernel(svar,sigma,x,t);

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




