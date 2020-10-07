import numpy as np
import OCGP
from sklearn import metrics
from scipy.io import loadmat

def processDrug(mypath,kernel,scoreTypes):

    data = loadmat(mypath+"dataset.mat")

    X_train  = data["X_Train"]
    X_test  = data["X_Test"]
    Y_train  = data["Y_Train"]
    Y_test  = data["Y_Test"]

    data = loadmat(mypath+"sel_features.mat")
    sel_features = data["sel_features"][0]
    sel_features = sel_features-1
    X_train = X_train[:, sel_features]
    X_test = X_test[:, sel_features]

    ocgp = OCGP.OCGP()
    svar = 0.0045

    if kernel == "adaptive":
        p = 30
        ls = ocgp.adaptiveHyper(X_train,p)
        ls = np.log(ls)
        X_train, X_test = ocgp.preprocessing(X_train, X_test, "minmax",True)
        ocgp.adaptiveKernel(X_train, X_test, ls, svar)
    elif kernel == "scaled":
        v = 0.8
        N = 4
        X_train, X_test = ocgp.preprocessing(X_train, X_test, "minmax",True)
        meanDist_xn, meanDist_yn = ocgp.scaledHyper(X_train, X_test, N)
        ocgp.scaledKernel(X_train, X_test, v, meanDist_xn, meanDist_yn, svar)

    print(kernel)
    for scoreType in scoreTypes:
        scores = ocgp.getGPRscore(scoreType)
        fpr, tpr, thresholds = metrics.roc_curve(Y_test, scores)
        AUC = metrics.auc(fpr, tpr)
        print(scoreType, ":", round(AUC, 4))

path = './DataDrugTarget/'

kernels = ['adaptive', 'scaled']
scores = ['mean', 'var','pred','ratio']

for kernel in kernels:
    processDrug(path, kernel,scores)