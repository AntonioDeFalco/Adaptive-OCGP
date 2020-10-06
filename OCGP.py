import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

class OCGP():

    def __init__(self):
        self.K = []
        self.Ks = []
        self.Kss = []
        self.L = []
        self.alpha = []
        self.v = []
        self.var = []

    def GPR_OCC(self):

        noise = 0.01
        self.K = self.K + noise * np.eye(np.size(self.K,0),np.size(self.K,1))
        self.Kss = self.Kss + noise * np.ones((np.size(self.Kss, 0),np.size(self.Kss, 1)))

        self.L = np.linalg.cholesky(self.K)
        self.alpha = np.linalg.solve(np.transpose(self.L),(np.linalg.solve(self.L,np.ones((np.size(self.K,0),1)))))

        #np.linalg.solve(B,b)
        #np.linalg.lstsq(B,b)

    def getGPRscore(self, modes):

        if modes == 'mean':
            score = np.dot(np.transpose(self.Ks),self.alpha)

        elif modes == 'var':
            if np.size(self.v) == 0:
                self.v = np.linalg.solve(self.L, self.Ks)
            score = [a + b for a, b in zip(-self.Kss, sum(np.multiply(self.v, self.v)))]

        elif modes == 'pred':
            if np.size(self.v) == 0:
                self.v = np.linalg.solve(self.L, self.Ks)
            #var = self.Kss - np.transpose(sum(np.multiply(v, v)))
            if np.size(self.var) == 0:
                self.var = [a - b for a, b in zip(self.Kss, sum(np.multiply(self.v, self.v)))]
            score = -0.5 * (np.divide(np.power((np.ones((np.size(self.var, 0), 1))) - (np.dot(np.transpose(self.Ks), self.alpha)), 2), self.var) + np.log(np.multiply(2 * np.pi , self.var)))

        elif modes == 'ratio':
            if np.size(self.v) == 0:
                self.v = np.linalg.solve(self.L, self.Ks)
            if np.size(self.var) == 0:
                self.var = [a - b for a, b in zip(self.Kss, sum(np.multiply(self.v, self.v)))]
            score = np.log(np.divide(np.dot(np.transpose(self.Ks),self.alpha),np.sqrt(self.var)))

        return score

    def seKernel(self,x,y,ls):

        #svar = 0.000045
        svar = 0.000045
        self.K = svar * np.exp(-0.5 * self.euclideanDistance(x, x)/ls)
        self.Ks = svar * np.exp(-0.5 * self.euclideanDistance(x, y)/ls)
        self.Kss = svar * np.ones((np.size(y, 0), 1))
        self.GPR_OCC()

    def adaptiveKernel(self,x,y,p,ls):

        #svar = 0.000045
        svar = 0.0045
        self.K = svar * np.exp(-0.5 * self.euclideanDistanceAdaptive(x, x, ls))
        self.K = (self.K + np.transpose(self.K))/2
        self.Ks = svar * np.exp(-0.5 * self.euclideanDistanceAdaptive(x, y, ls))
        self.Kss = svar * np.ones((np.size(y, 0), 1))
        self.GPR_OCC()

    def scaledKernel(self,x,y,v,N,meanDist_xn,meanDist_yn):

        #svar = 0.000045
        svar = 0.0045
        self.K = svar * np.exp(-0.5 * self.euclideanDistanceScaled(x, x, v, meanDist_xn,meanDist_xn))
        self.Ks = svar * np.exp(-0.5 * self.euclideanDistanceScaled(x, y, v, meanDist_xn,meanDist_yn))
        self.Kss = svar * np.ones((np.size(y, 0), 1))
        self.GPR_OCC()

    def euclideanDistanceScaled(self,x, y, v, meanDist_xn, meanDist_yn):
        distmat = np.zeros((np.size(x, 0), np.size(y, 0)))
        for i in range(0, np.size(x, 0)):
            for j in range(0,np.size(y, 0)):
                dist = (x[i, :] - y[j, :])
                dist2 = np.dot(dist,dist)
                dist = np.sqrt(dist2)
                epsilon_ij = (meanDist_xn[i] + meanDist_yn[j] + dist) / 3
                buff = np.divide(dist2 , (v * epsilon_ij))
                distmat[i, j] = buff
        return distmat

    def euclideanDistanceAdaptive(self,x, y, ls):
        distmat = np.zeros((np.size(x, 0), np.size(y, 0)))
        for i in range(0, np.size(x, 0)):
            for j in range(0,np.size(y, 0)):
                buff = (x[i,:] - y[j,:])
                buff = buff / ls[i]
                distmat[i, j] = np.dot(buff, buff)
        return distmat

    def euclideanDistance(self,x, y):
        distmat = np.zeros((np.size(x, 0), np.size(y, 0)))
        for i in range(0, np.size(x, 0)):
            for j in range(0,np.size(y, 0)):
                buff = (x[i,:] - y[j,:])
                distmat[i, j] = np.dot(buff, buff)
        return distmat

    def knn(self, data, k):
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(data)
        dist = neigh.kneighbors(data, k)
        return dist[0]

    def adaptiveHyper(self,x,p):
        dist = self.knn(x,p)
        ls = dist[:, p - 1]
        return ls

    def scaledHyper(self,x,y,N):
        dist_xn = self.knn(x, N)
        #dist_yn = self.knn(y,N)
        #dist_yn = distance.cdist(x, y)
        dist_yn = distance.cdist(y, x) #(as MATLAB)
        dist_yn = np.sort(dist_yn, axis=1)
        dist_yn = dist_yn[:, 0:N]
        meanDist_xn = np.mean(dist_xn,1)
        meanDist_yn = np.mean(dist_yn,1)
        return meanDist_xn, meanDist_yn

    def preprocessing(self,x,y,mode,pca = False):

        if mode == "minmax":
            scaler = MinMaxScaler()
            all = np.vstack([x, y])
            scaler.fit(all)
            all = scaler.transform(all)
            x = all[0: np.size(x,0),:]
            y = all[np.size(x,0):np.size(all),:]
        elif mode == "zscore":
            scaler = StandardScaler()
            scaler.fit(x)
            x = scaler.transform(x)
            y = scaler.transform(y)

        if pca == True:
            all = np.vstack([x, y])
            pca = PCA(n_components=0.80)
            all = pca.fit_transform(all)
            x = all[0: np.size(x,0),:]
            y = all[np.size(x,0):np.size(all),:]

        return x,y