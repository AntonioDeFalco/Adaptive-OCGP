from SVDD.src.svdd import SVDD
from scipy.io import loadmat
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

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

    if pca == True:
        all = np.vstack([x, y])
        pca = PCA(n_components=0.80)
        all = pca.fit_transform(all)
        x = all[0: np.size(x, 0), :]
        y = all[np.size(x, 0):np.size(all), :]

    return x, y

path = './DataDrugTarget/'
data = loadmat(path + "dataset.mat")

X_train = data["X_Train"]
X_test = data["X_Test"]
Y_train = data["Y_Train"]
Y_test = data["Y_Test"]

"""
data = loadmat(path + "sel_features.mat")
sel_features = data["sel_features"][0]
sel_features = sel_features - 1
X_train = X_train[:, sel_features]
X_test = X_test[:, sel_features]
"""

mode = "minmax"
X_train, X_test = preprocessing(X_train, X_test, mode)

#trainLabel = np.transpose(np.matrix(trainLabel))


#testLabel = np.transpose(np.matrix(testLabel))

# set SVDD parameters
parameters = {"positive penalty": 0.9,
              "negative penalty": [],
              "kernel": {"type": 'ploy', "degree": 3, "offset": 0},
              "option": {"display": 'off'}}

#"kernel": {"type": 'ploy', "degree": 3, "offset": 0},
#"kernel": {"type": 'tanh', "gamma": 1e-4, "offset": 0},
#"kernel": {"type": 'gauss', "width": 1 / 80},

# construct an SVDD model
svdd = SVDD(parameters)

# train SVDD model
svdd.train(X_train, Y_train)

# test SVDD model
distance, accuracy = svdd.test(X_test, Y_test)

fpr, tpr, thresholds = metrics.roc_curve(Y_test, -distance)
AUCs = metrics.auc(fpr, tpr)

print(AUCs)
