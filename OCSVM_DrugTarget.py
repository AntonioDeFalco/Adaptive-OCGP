from scipy.io import loadmat
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preprocessing(x, y, mode, pca=False):
    if mode == "minmax":
        scaler = MinMaxScaler()
        all = np.vstack([x, y])
        scaler.fit(all)
        all = scaler.transform(all)
        x = all[0: np.size(x, 0), :]
        y = all[np.size(x, 0):np.size(all), :]
    elif mode == "zscore":
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)
        y = scaler.transform(y)

    return x, y

path = './DataDrugTarget/'
data = loadmat(path + "dataset.mat")

X_train = data["X_Train"]
X_test = data["X_Test"]
Y_train = data["Y_Train"]
Y_test = data["Y_Test"]

mode = "minmax"
X_train, X_test = preprocessing(X_train, X_test, mode)

ocsvm_clf = OneClassSVM(gamma='auto', kernel='rbf', nu=0.01).fit(X_train)
y_pred = ocsvm_clf.predict(X_test)

fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred)
AUC = metrics.auc(fpr, tpr)

print(AUC)