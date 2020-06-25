# Drug Target - Gaussian Process OCC

This repository contains an implementation of Gaussian Processes applied to a One Class Classification (OCC) problem, starting from Kemmler et al. [1].

Since the OCC problem does not allow the automatic selection of GP hyperparameters, we propose two kernels that use not a hyperparameter with a fixed value but an adaptive hyperparameter, which varies for each sample and is automatically selected according to the distance of the training samples.
* *Adaptive Kernel* the hyperparameter is based on the distance of the training sample from its k-th nearest training sample
* *Scaled Kernel* combines the distance between the samples with the average distance of the samples from their nearest.

The proposed kernels are compared with an implementation (***hyperparameter_Selection.m***) of the best known method for the hyperparameter selection for OCC of Xiao et al. [2]. 
The main problem addressed is the selection and prioritization of drug targets, the ***OCC.m*** script contains testing on DrugTarget dataset with which we get AUC 0.90.

# Script Options 

###### Preprocessing
* ***logtrasform***:                boolean value perform log transform of features with heavy-tailed distribution
* ***scale***:                      boolean value perform min-max normalization
* ***norm_zscore***:                boolean value perform z-score normalization
* ***pca_exc***:                    boolean value perform PCA 
* ***perc_pca***:                   variance percentage

###### Feature Selection
* ***sparse_selection***:   boolean value perform Sparse Features Selection
* ***exec_SFS***:           boolean value perform Sequential forward selection (SFS) 
* ***exec_SBS***:           boolean value perform Sequential Backward Selection (SBS) 
* ***score_mode***:         sequential selection criterion ('mean' or 'var')
* ***load_featuresSFS***:    boolean value load features selected with SFS

###### Covariance Function
* ***distance_mode***
    * *'euclidean'*: Use Euclidean distance    
    * *'pearson'*:   Use Pearson distance (1- Pearson correlation coefficient) 

* ***data_process***
    * *'before'*:   Preprocessing data before computing hyperparameter 
    * *'after'*:    Preprocessing data after computing hyperparameter 

* ***kernel***
    * *'scaled'*:         Scaled kernel
    * *'adaptive'*:       Adaptive kernel
    * *'hyperOCC'*:       Hyperparameter Selection of Xiao et al.

###### Adaptive kernel
* ***log_sigma***:           boolean value perform log transform of hyperparameters     
* ***k_adapt***:             k parameter of Adaptive kernel

###### Scaled kernel
* ***k_scaled***:            k parameter of Scaled kernel usually (10-30)
* ***mu_scaled***:           hyperparameter, usually (0.3-0.8)

# UCI Benchmars 

The ***datasetUCI.m*** script contains testing on UCI datasets downloaded from [url](http://homepage.tudelft.nl/n9d04/occ/index.html).
The following table shows AUC scores for mean and negative variance on UCI datasets. At left the results with Xiao et al. hyperparameter selection, on the right the results of our proposed method.

|Dataset    |    Xiao(mean)   |    Xiao(-var)   |  Adaptive(mean) | Adaptive(-var)  |   Scaled(mean)  |   Scaled(-var)  |
|-----------|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|
| Abalone   |    **0,7894**   |    **0,7897**   |      0,7745     |      0,7428     |      0,7742     |      0,7092     |
| Balance   |      0,8366     |      0,8735     | 	**0,9468**    | 	**0,9682**  |      0,8657     |      0,9402     |
| Biomed    |      0,8998     |      0,9036     |      0,9028     |      0,8960     | 	**0,9073**    | 	**0,9117**  |
| Heart     |      0,8339     | 	**0,8379**  |      0,8093     |      0,7925     | 	**0,8408**    |      0,8135     |
| Hepatitis | 	  **0,8378**  | 	**0,8379**  |      0,8006     |      0,7794     |      0,8242     |      0,7963     |
| Housing   |      0,7917     |      0,7874     | 	**0,8677**    | 	**0,8680**  |      0,8107     |      0,8492     |
| Ionosphere|      0,9265     |      0,9504     |      0,9550     |      0,9649     | 	**0,9697**    | 	**0,9712**  |
| Vehicle   |      0,5183     |      0,5714     | 	**0,7965**    | 	**0,8656**  |      0,6855     |      0,8187     |
| Waveform  |      0,7497     |      0,8004     |      0,7808     |     **0,8167**  |    **0,8024**   |      0,7998     |
|**AVERAGE**|      0,7982     |      0,8169     | 	**0,8482**    | 	**0,8549**  |      0,8312     |      0,8455     |


# 1-D OCC 

The ***oneD_demo.m*** script contains a demo of the comparison in the case of a one-dimensional OCC setting, with a fixed hyperparameter and our proposed kernels.

![1-D example](https://github.com/AntonioDeFalco/DrugTarget-GPOCC/blob/master/all1D.png?raw=true)

# Cite:

[1] Michael Kemmler and Erik Rodner and Joachim Denzler: "One-Class Classification with Gaussian Processes", Proceedings of the 10th Asian Conference on Computer Vision, 2010.

[2] Yingchao Xiao, Huangang Wang, and Wenli Xu: "Hyperparameter Selection for Gaussian Process One-Class Classification", IEEE Transactions on Neural Networks and Learning Systems, 2015.

[3] E. Elhamifar, G. Sapiro, and R. Vidal: "Sparse Modeling for Finding Representative Objects", IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012.

[4] Brian Moore (2020). PCA and ICA Package (https://www.mathworks.com/matlabcentral/fileexchange/38300-pca-and-ica-package), MATLAB Central File Exchange. Retrieved June 25, 2020.