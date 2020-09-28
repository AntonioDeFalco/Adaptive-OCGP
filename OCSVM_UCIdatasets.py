from os import listdir
from os.path import isfile, join
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

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

    for i in range(0,20):
        X_train, X_test, y_train, y_test = train_test_split(class1, np.ones(np.size(class1,0)), test_size=0.20)
        y_test = np.concatenate([np.ones(np.size(X_test, 0)), np.zeros(np.size(class2, 0))])
        X_test = np.vstack([X_test, class2])

        ocsvm_clf = OneClassSVM(gamma='auto', kernel='rbf', nu=0.1).fit(X_train)
        x_pred = ocsvm_clf.predict(X_train)
        y_pred = ocsvm_clf.predict(X_test)

        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
        AUCs = np.append(AUCs,metrics.auc(fpr, tpr))

    print(name)
    print(np.mean(AUCs))
    AUCmeans = np.append(AUCmeans, np.mean(AUCs))

print("Average")
print(np.mean(AUCmeans))