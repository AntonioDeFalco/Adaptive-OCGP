
install.packages("gelnet")
install.packages("R.matlab")
install.packages("pROC")
install.packages("pspearman")

install.packages("dgof")

require(pspearman)
require(pROC)
require(gelnet)
require(R.matlab)
require(dgof)

#file <- './UCI_OCC_DATASETS/balance.mat'

files <- list.files(path="./UCI_OCC_DATASETS", pattern="*.mat", full.names=TRUE, recursive=FALSE)

meanAUCS <- c()  

for (file in files){

  class1 <- c()
  class2 <- c()
  AUCs <- c()
  
  data <- readMat(file)
  
  dataset <- data[['x']]
  
  for(i in 1:20) {
    
    scores <- c()
    
    for (i in 1:dim(dataset[['nlab']])[1]){
      if (dataset[['nlab']][i] == 1){
        class1 <- rbind(class1, dataset[['data']][i,])
      }
      else{
        class2 <- rbind(class2, dataset[['data']][i,])
      }
    }
    
    numSampl <- c(dim(class1)[1],dim(class2)[1])
    
    I <- which(numSampl==max(numSampl))
    
    if (I == 2){
      temp <- class1;
      class1 <-  class2;
      class2 <- temp;
    }
    
    train_index <- sample(1:nrow(class1), 0.8 * nrow(class1))
    test_index <- setdiff(1:nrow(class1), train_index)
    
    x <- class1[train_index,]
    
    t <- class1[test_index,]
    t <- rbind(t, class2)
    
    training_zscore <- scale(x)
    validation_zscore <- scale(t, center = colMeans(x), scale = apply(x, 2, sd))
    
    y <- c(rep(1, dim(x)[1]))
    t_label <- c(rep(1, length(test_index)), rep(0, dim(class2)[1]))
    
    model <- gelnet(x, NULL, 0, 1)
    
    # for(i in 1:dim(t)[1]) {
    #   
    #   #out1 <- cor.test(model$w,t[1,],method="pearson")
    #   #out1 <- spearman.test(model$w, t[i,])
    #   
    #   out1 <- cor.test(model$w,t[i,],method="spearman")
    #   score <- out1$statistic
    #   scores <- c(scores,score)
    #   
    # }
    
    scores <- exp(t %*% model$w)/(1+ exp(t %*% model$w))
    
    roc_obj <- roc(t_label, scores)
    AUC <- auc(roc_obj)
    AUCs <- c(AUCs,AUC)
    print(AUC)
    
  }
  
  print(file)
  
  meanAUCS <- c(meanAUCS,mean(AUCs)) 
  
  print(mean(AUCs))
  
}

dataf <- data.frame (files,meanAUCS)
write.table(dataf, file = "./meanAucs.csv", sep = "\t", quote = FALSE, col.names = FALSE)
