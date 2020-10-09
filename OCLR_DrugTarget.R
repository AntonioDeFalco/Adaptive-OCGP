# Description: One Class Logistic Regression on UCI Datasets.
#
# Author: Antonio De Falco

#install.packages("gelnet")
#install.packages("R.matlab")

require(gelnet)
require(R.matlab)

norm_minmax <- function(x){
    (x- min(x)) /(max(x)-min(x))
}

file <- "./DataDrugTarget/dataset.mat"
data <- readMat(file)

X_train <- data[["X.Train"]]
X_test <- data[["X.Test"]]
Y_train <- as.vector(data[["Y.Train"]])
Y_test <- as.vector(data[["Y.Test"]])

#Min-Max Normalization
maxs <- apply(rbind(X_train,X_test), 2, max)
mins <- apply(rbind(X_train,X_test), 2, min)
X_train <- scale(X_train, center = mins, scale = maxs - mins)
X_test <- scale(X_test, center = mins, scale = maxs - mins)

model <- gelnet(X_train, NULL, 0, 1)

scores <- as.vector(exp(X_test %*% model$w)/(1+ exp(X_test %*% model$w)))

roc_obj <- roc(Y_test, scores)
AUC <- auc(roc_obj)
print(AUC)