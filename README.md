# Drug Target - Gaussian Process OCC

This repository contains an implementation of Gaussian Processes applied to a One Class Classification (OCC) problem, starting from Kemmler et al. [1].
A new method is proposed for selecting the lenght-scale hyperparameter in the SE kernel obtaining an adaptive kernel. It is compared with the only proposed method for the hyperparameter selection for OCC of Xiao et al. [2]. 
The main problem addressed is the selection and prioritization of drug targets, the OCC.m script contains testing on DrugTarget dataset with which we get AUC 0.90, but we confirm the validity of the proposed method on the following datasets.

# UCI Benchmars 

The datasetUCI.m script contains testing on UCI datasets downloaded from [url](http://homepage.tudelft.nl/n9d04/occ/index.html)

AUC scores for mean and variance on UCI datasets. At left the results with Xiao et al. hyperparameter selection, on the right the results of our proposed method.

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

The oneD_demo.m script contains a demo of the comparison in the case of a one-dimensional OCC setting, with a fixed hyperparameter and our adaptive kernel.

# Cite:

[1] Michael Kemmler and Erik Rodner and Joachim Denzler: "One-Class Classification with Gaussian Processes", Proceedings of the 10th Asian Conference on Computer Vision, 2010.

[2] Yingchao Xiao, Huangang Wang, and Wenli Xu: "Hyperparameter Selection for Gaussian Process One-Class Classification", IEEE Transactions on Neural Networks and Learning Systems, 2015.