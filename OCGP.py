import numpy as np

class OCGP():

    def __init__(self):
        self.K = []
        self.Ks = []
        self.Kss = []
        self.L = []
        self.alpha = []

    def GPR_OCC(self):

        noise = 0.01
        self.K = self.K + noise * np.eye(np.size(self.K,0),np.size(self.K,1))
        self.Kss = self.Kss + noise * np.ones((np.size(self.Kss,0),np.size(self.Kss,1)))

        self.L = np.linalg.cholesky(self.K)
        self.alpha = np.linalg.solve(np.transpose(self.L),(np.linalg.solve(self.L,np.ones((np.size(self.K,0),1)))))

        #np.linalg.solve(B,b)
        #np.linalg.lstsq(B,b)

    def getGPRscore(self, modes):

        if modes == 'mean':
            score = np.transpose(self.Ks) * self.alpha

        elif modes == 'var':
            v = np.linalg.solve(self.L, self.Ks)
            score = -self.Kss + np.transpose(sum(np.multiply(v, v)))

        elif modes == 'pred':
            v = np.linalg.solve(self.L, self.Ks)
            var = self.Kss - np.transpose(sum(np.multiply(v, v)))
            score = -0.5 * (np.divide((np.ones((np.size(var, 0), 1)) - np.power((np.transpose(self.Ks) * self.alpha), 2)), var + np.log(2 * np.pi * var)))

        elif modes == 'ratio':
            v = np.linalg.solve(self.L, self.Ks)
            score = np.divide(np.log((np.transpose(self.Ks) * self.alpha),np.sqrt(self.Kss - np.transpose(sum(np.multiply(v, v))))))

        return score

    def adaptiveKernel(self,svar,ls,x,y):
        self.K = svar * np.exp(-0.5 * self.euclideanDistance(x, x, ls))
        self.K = (self.K + np.transpose(self.K))/2
        self.Ks = svar * np.exp(-0.5 * self.euclideanDistance(x, y, ls))
        self.Kss = svar * np.ones((np.size(y, 0), 1))
        self.GPR_OCC()

    def euclideanDistance(self,x, y, ls):
        distmat = np.zeros((np.size(x, 0), np.size(y, 0)))
        for i in range(0, np.size(x, 0)):
            for j in range(0,np.size(y, 0)):
                buff = (x[i,:] - y[j,:])
                #buff = buff / ls[i]
                distmat[i, j] = np.dot(buff, buff)
        return distmat
    """"
    def knn(self):
        [idx, dist] = knnsearch(x, x, 'k', k_adapt) #%, 'Distance', 'jaccard');
        if log_sigma
        sigma = log(dist(:,k_adapt));
        else
        sigma = dist(:, k_adapt);

    def preprocessing(self):
    """
