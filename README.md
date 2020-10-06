# Adaptive One-Class Gaussian Processes - Prioritization of Oncology Drug Targets

This repository contains a Python implementation of Adaptive One-Class Gaussian Processes, starting from OCGP Kemmler et al. [1].

Since the OCC problem does not allow the automatic selection of GP hyperparameters, we propose two kernels that use not a hyperparameter with a fixed value but an adaptive hyperparameter, which varies for each sample and is automatically selected according to the distance of the training samples.
* *Adaptive Kernel* the hyperparameter is based on the distance of the training sample from its k-th nearest training sample
* *Scaled Kernel* combines the distance between the samples with the average distance of the samples from their nearest.

The proposed kernels are compared with an implementation of the best known method for the hyperparameter selection for OCC of Xiao et al. [2]. 
The main problem addressed is the selection and prioritization of drug targets.

# UCI Benchmars 

The ***datasetUCI.m*** script contains testing on UCI datasets downloaded from [url](http://homepage.tudelft.nl/n9d04/occ/index.html).
The following table shows AUC scores for mean and negative variance on UCI datasets, comparing the proposed kernels and implementation of the hyperparameter selection of Xiao et al. [2].

|Dataset   |    Xiao(mean)  |  Xiao(-var) |  Adapt.(mean) | Adapt.(-var)  |   Scaled(mean)  |   Scaled(-var)  |
|----------|:--------------:|:------------:|:---------------:|:---------------:|:---------------:|:---------------:|
|Abalone   |    **0,7894**  |   **0,7897**|      0,7745     |      0,7428     |      0,7742     |      0,7092     |
|Balance   |      0,8366    |      0,8735 | 	**0,9468**    | 	**0,9682**  |      0,8657     |      0,9402     |
|Biomed    |      0,8998    |      0,9036 |      0,9028     |      0,8960     | 	**0,9073**    | 	**0,9117**  |
|Heart     |      0,8339    | 	**0,8379** |      0,8093     |      0,7925     | 	**0,8408**    |      0,8135     |
|Hepatitis | 	  **0,8378**| 	**0,8379** |      0,8006     |      0,7794     |      0,8242     |      0,7963     |
|Housing   |      0,7917    |      0,7874 | 	**0,8677**    | 	**0,8680**  |      0,8107     |      0,8492     |
|Ionosphere|      0,9265    |      0,9504 |      0,9550     |      0,9649     | 	**0,9697**    | 	**0,9712**  |
|Vehicle   |      0,5183    |      0,5714 | 	**0,7965**    | 	**0,8656**  |      0,6855     |      0,8187     |
|Waveform  |      0,7497    |      0,8004 |      0,7808     |     **0,8167**  |    **0,8024**   |      0,7998     |
|**AVERAGE**|      0,7982   |      0,8169 | 	**0,8482**    | 	**0,8549**  |      0,8312     |      0,8455     |


# Cite:

[1] Michael Kemmler and Erik Rodner and Joachim Denzler: "One-Class Classification with Gaussian Processes", Proceedings of the 10th Asian Conference on Computer Vision, 2010.

[2] Yingchao Xiao, Huangang Wang, and Wenli Xu: "Hyperparameter Selection for Gaussian Process One-Class Classification", IEEE Transactions on Neural Networks and Learning Systems, 2015.

