from src.svdd import SVDD
from src.visualize import Visualization as draw
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import numpy as np
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
    for i in range(0, 20):
        trainData, testData, trainLabel, testLabel = train_test_split(class1, np.ones(np.size(class1, 0)), test_size=0.20)
        testLabel = np.concatenate([np.ones(np.size(testData, 0)), np.zeros(np.size(class2, 0))-1])
        testData = np.vstack([testData, class2])

        trainLabel = np.transpose(np.matrix(trainLabel))
        #testLabel = np.transpose(np.matrix(testLabel))

        # set SVDD parameters
        parameters = {"positive penalty": 0.9,
                      "negative penalty": [],
                      "kernel": {"type": 'ploy', "degree": 3},
                      "option": {"display": 'off'}}

        #"kernel": {"type": 'ploy', "degree": 2, "offset": 0},
        #"kernel": {"type": 'tanh', "gamma": 1e-4, "offset": 0},
        #"kernel": {"type": 'gauss', "width": 1 / 80},

        # construct an SVDD model
        svdd = SVDD(parameters)

        # train SVDD model
        svdd.train(trainData, trainLabel)

        # test SVDD model
        distance, accuracy = svdd.test(testData, testLabel)

        fpr, tpr, thresholds = metrics.roc_curve(testLabel, -distance)
        AUCs = np.append(AUCs,metrics.auc(fpr, tpr))

    print(name)
    print(np.mean(AUCs))
    AUCmeans = np.append(AUCmeans, np.mean(AUCs))

print("Average")
print(np.mean(AUCmeans))
    # visualize the results
    #draw.testResult(svdd, distance)
    #draw.testROC(testLabel, distance)