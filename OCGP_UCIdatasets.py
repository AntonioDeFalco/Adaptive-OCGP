import numpy as np
import OCGP
from os import listdir
from os.path import isfile, join
from sklearn import metrics
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

mypath = '/Users/antonio/Desktop/UCIdatasetpy/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles.sort()

AUCmeans = np.array([])

for name in onlyfiles:

    data = loadmat(mypath+name)
    class1 = data["class1"]
    class2 = data["class2"]

    standard_scaler = StandardScaler()
    standard_scaler.fit(class1)
    class1 = standard_scaler.transform(class1)
    class2 = standard_scaler.transform(class2)

    AUCs = np.array([])

    for i in range(0, 20):
        X_train, X_test, y_train, y_test = train_test_split(class1, np.ones(np.size(class1,0)), test_size=0.20)
        y_test = np.concatenate([np.ones(np.size(X_test, 0)), np.zeros(np.size(class2, 0))])
        X_test = np.vstack([X_test, class2])

        ocgp = OCGP.OCGP()

        p = 2
        ocgp.adaptiveKernel(X_train,X_test,p)

        modes = ['mean', 'var', 'pred', 'ratio']

        #for i in range(0,1):
        i = 1
        score = ocgp.getGPRscore(modes[i])
        fpr, tpr, thresholds = metrics.roc_curve(y_test, score)
        AUCs = np.append(AUCs,metrics.auc(fpr, tpr))

    print(name)
    print(np.mean(AUCs))
    AUCmeans = np.append(AUCmeans, np.mean(AUCs))

print("Average")
print(np.mean(AUCmeans))