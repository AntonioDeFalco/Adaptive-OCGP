import numpy as np
import OCGP
from os import listdir
from os.path import isfile, join
from sklearn import metrics
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def processUCI(mypath,kernel,score,id,name,AUCmeans):

    data = loadmat(mypath+name)
    class1 = data["class1"]
    class2 = data["class2"]

    standard_scaler = StandardScaler()
    standard_scaler.fit(class1)
    class1 = standard_scaler.transform(class1)
    class2 = standard_scaler.transform(class2)

    numIter = 20

    AUCs = np.zeros(numIter)

    for id2 in range(0, numIter):
        getAUC_OCGP_UCI(class1, class2, kernel, score, id2, AUCs)

    print ("Completed Iterations of:",name)
    AUCmeans[id] = np.mean(AUCs)


def getAUC_OCGP_UCI(posSamples, negSamples, kernel,score,id,AUCs):

    X_train, X_test, y_train, y_test = train_test_split(posSamples, np.ones(np.size(posSamples, 0)), test_size=0.20, random_state = id)
    y_test = np.concatenate([np.ones(np.size(X_test, 0)), np.zeros(np.size(negSamples, 0))])
    X_test = np.vstack([X_test, negSamples])

    ocgp = OCGP.OCGP()
    svar = 0.000045
    if kernel == "adaptive":
        p = 2
        X_train, X_test = ocgp.preprocessing(X_train, X_test, "zscore")
        ls = ocgp.adaptiveHyper(X_train, p)
        ocgp.adaptiveKernel(X_train, X_test, ls, svar)
    elif kernel == "scaled":
        v = 0.8
        N = 5
        X_train, X_test = ocgp.preprocessing(X_train, X_test, "zscore")
        meanDist_xn, meanDist_yn = ocgp.scaledHyper(X_train, X_test, N)
        ocgp.scaledKernel(X_train, X_test, v, meanDist_xn, meanDist_yn, svar)

    scores = ocgp.getGPRscore(score)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, scores)
    AUC = metrics.auc(fpr, tpr)
    AUCs[id] = AUC

mypath = './UCIdatasetpy/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles.sort()

numDatasets = np.size(onlyfiles)

AUCmeans = np.zeros(numDatasets)

kernels = ['adaptive', 'scaled']
scores = ['mean', 'var']

for kernel in kernels:
    for score in scores:
        for id in range(0,np.size(onlyfiles)):
            processUCI(mypath, kernel, score, id, onlyfiles[id], AUCmeans)

        print(kernel)
        print(score)
        print("datasets processing complete.")
        print(AUCmeans)

        print("Average")
        print(np.mean(AUCmeans))
