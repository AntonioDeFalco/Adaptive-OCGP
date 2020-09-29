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

name = onlyfiles[0]

data = loadmat(mypath+name)
class1 = data["class1"]
class2 = data["class2"]

standard_scaler = StandardScaler()
standard_scaler.fit(class1)
class1 = standard_scaler.transform(class1)
class2 = standard_scaler.transform(class2)

X_train, X_test, y_train, y_test = train_test_split(class1, np.ones(np.size(class1,0)), test_size=0.20)
y_test = np.concatenate([np.ones(np.size(X_test, 0)), np.zeros(np.size(class2, 0))])
X_test = np.vstack([X_test, class2])

ocgp = OCGP.OCGP()

#x = np.array([[0.25,0.47,0.57,0.98,0.77,0.22],[0.22,0.88,0.55,0.44,0.22,0.01]])
#y = np.array([[0.40,0.54,0.67,0.65,0.44,0.25],[0.42,0.55,0.88,0.77,0.44,0.11]])

#x = np.random.rand(2, 3)
#y = np.random.rand(2, 3)
ls = np.random.rand(np.size(X_train, 0))

svar = 0.000045;

ocgp.adaptiveKernel(svar,ls,X_train,X_test)

modes = ['mean', 'var', 'pred', 'ratio']
score = ocgp.getGPRscore(modes[0])
print(score)

fpr, tpr, thresholds = metrics.roc_curve(y_test, score)
print(metrics.auc(fpr, tpr))
