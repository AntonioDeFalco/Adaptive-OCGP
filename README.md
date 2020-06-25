# Drug Target - Gaussian Process OCC

This repository contains an implementation of Gaussian Processes applied to a One Class Classification (OCC) problem, starting from Kemmler et al. [1].

Since the OCC problem does not allow the automatic selection of GP hyperparameters, we propose two kernels that use not a hyperparameter with a fixed value but an adaptive hyperparameter, which varies for each sample and is automatically selected according to the distance of the training samples.
* Adaptive Kernel the hyperparameter is based on the distance of the training sample from its k-th nearest training sample
* Scaled Kernel combines the distance between the samples with the average distance of the samples from their nearest.

The proposed kernels are compared with an implementation (hyperparameter_Selection.m) of the best known method for the hyperparameter selection for OCC of Xiao et al. [2]. 
The main problem addressed is the selection and prioritization of drug targets, the OCC.m script contains testing on DrugTarget dataset with which we get AUC 0.90.

# UCI Benchmars 

The datasetUCI.m script contains testing on UCI datasets downloaded from [url](http://homepage.tudelft.nl/n9d04/occ/index.html).
The following table shows AUC scores for mean and variance on UCI datasets. At left the results with Xiao et al. hyperparameter selection, on the right the results of our proposed method.

| Dataset    |XiaoSelection[2] (mean)|XiaoSelection[2] (variance)| AdaptiveKernel (mean)| AdaptiveKernel (variance)|
|------------|:---------------------:|:-------------------------:|:--------------------:|:------------------------:|
| Abalone    |         0,7894        |           0,7897          |        0,7745        |          0,7428          |
| Balance    |         0,8366        |           0,8735          |        0,9468        |          0,9682          |
| Biomed     |         0,8998        |           0,9036          |        0,9028        |          0,8960          |
| Heart      |         0,8339        |           0,8379          |        0,8093        |          0,7925          |
| Hepatitis  |         0,8378        |           0,8379          |        0,8006        |          0,7794          |
| Housing    |         0,7917        |           0,7874          |        0,8677        |          0,8680          |
| Ionosphere |         0,9265        |           0,9504          |        0,9550        |          0,9649          |
| Vehicle    |         0,5183        |           0,5714          |        0,7965        |          0,8656          |
| Waveform   |         0,7497        |           0,8004          |        0,7808        |          0,8167          |
| AVERAGE    |         0,7982        |           0,8169          |        0,8482        |          0,8549          |

# 1-D OCC 

The oneD_demo.m script contains a demo of the comparison in the case of a one-dimensional OCC setting, with a fixed hyperparameter and our proposed kernels.

# Cite:

[1] Michael Kemmler and Erik Rodner and Joachim Denzler: "One-Class Classification with Gaussian Processes", Proceedings of the 10th Asian Conference on Computer Vision, 2010.

[2] Yingchao Xiao, Huangang Wang, and Wenli Xu: "Hyperparameter Selection for Gaussian Process One-Class Classification", IEEE Transactions on Neural Networks and Learning Systems, 2015.