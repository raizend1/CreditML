rm(list = ls())
# Initialization ----------------------------------------------------------
# Needed libraries
library(ggplot2)
library(mice)
library(kernlab)
library(class)
library(e1071)
library(psych)
library(DMwR)
library(chemometrics)
library(ggrepel)
library(ggthemes)
library(robustbase)
library(knitcitations)
library(doParallel)
library(gridExtra)
library(DMwR)
library(pROC)
library(caret) 
library(e1071)
library(nnet)


# Initialisation ----------------------------------------------------------

# Read initial data
setwd('/Users/cesc/Documents/UPC MIRI-FIB/2Q/Machine Learning/Project/CreditML_Project/code/')
load('Environment_EDA.RData')

set.seed(555)

nd.indexes <- sample(which(credit$default.payment.next.month == "Not default"),6533)
d.indexes <- which(credit$default.payment.next.month == "Default")
subsample.index <- sort(c(nd.indexes,d.indexes))

table(credit[subsample.index,]$default.payment.next.month)

#balanced
credit.train <- sample(subsample.index, size = ceiling(length(subsample.index)*0.8))
credit.test <- subset(subsample.index, !(subsample.index %in% credit.train))

credit.x <- credit[credit.train,-24]
credit.y <- credit[credit.train,24]

# Final train dataset size
dim(credit.x)
# [1] 10453    23
length(credit.y)
# [1] 10453

# Logistic Regression -----------------------------------------------------
train <- cbind(credit.x, default = credit.y)
pay <- c(7,8,9,10,11) 
train <- train[,-pay]
( logReg <- glm(default ~ ., data = train, family = binomial(link=logit)) )
logReg.step <- step(logReg) # Warning: It takes a minute
summary(logReg.step)
logReg <- glm (logReg.step$formula, data = train, family = binomial(link = logit))
summary(logReg)


# Test error
test_aux <- credit[credit.test,]
test <- credit[credit.test,-24]

pred <- predict (
  object = logReg,
  newdata = test,
  type = 'response',
  na.action = na.pass
)

p <- 0.5
predictions <- NULL
predictions[pred >= p] <- 1
predictions[pred < p] <- 0

( tab <- table(Truth = test_aux$default.payment.next.month, Pred=predictions) )
(error.test <- 100*(1-sum(diag(tab))/nrow(test))) # 29.54 % of error as well.

predictions <- as.factor(predictions)
levels(predictions) <- c('Not default','Default')
confusionMatrix(predictions, test_aux$default.payment.next.month)

truth <- test_aux$default.payment.next.month

# Plot the ROC curve
library(pROC)
length(pred)
test
pred.lg <- prediction(as.numeric(pred), credit[credit.test,24])
roc1 <- performance(pred.lg,measure="fpr",x.measure="tpr")

# Support Vector Machine --------------------------------------------------
# The final values used for the model were sigma = 0.09374442 and C = 1.
credit.svm <- ksvm(default.payment.next.month ~.,
                   data=credit[credit.train,], kernel='rbfdot', type = "C-svc",
                   kpar=list(sigma=0.09374442),C=1,prob.model = TRUE)

pred.test.svm <- predict(credit.svm, newdata=credit[credit.test,],type="prob")
invisible(require(ROCR))
pred.svm <- prediction(as.data.frame(pred.test.svm)$`Not default`, credit$default.payment.next.month[credit.test])
roc3 <- performance(pred.svm,measure="tpr",x.measure="fpr")

# Neural Networks ---------------------------------------------------------
train <- cbind(credit.x, default = credit.y)
test <- credit[credit.test,]

model.nnet <- nnet( default ~ . ,
                    data = train,
                    size=10,
                    MaxNWts = 2000,
                    maxit=2000,
                    decay = 0.01584 )

(pred.test.nnet <- as.factor(predict(model.nnet, newdata = test[,-24], type = 'raw')))

# Plot the ROC curve
library(pROC)
length(pred.test.nnet)
names(test)
pred.nnet <- prediction(as.numeric(pred.test.nnet), test[,24])
roc2 <- performance(pred.nnet,measure="fpr",x.measure="tpr")

# Random Forest -----------------------------------------------------------
rf.train.control<- trainControl(method = "oob",savePred=TRUE)

cl <- makeCluster(detectCores())
registerDoParallel(cl)
start.time<-proc.time()
rf.tune <- train(x=credit.x[,-factor.indexes], y= credit.y,
                 method = "rf", trControl = rf.train.control, metric = "Accuracy")
end.time<-proc.time()
time.rf<- end.time-start.time
stopCluster(cl)

pred.test.rf <- predict(rf.tune, credit[credit.test,],type="prob")
invisible(require(ROCR))
pred.rf <- prediction(as.data.frame(pred.test.rf)$`Not default`, credit$default.payment.next.month[credit.test])
roc4 <- performance(pred.rf,measure="tpr",x.measure="fpr")


# Plot ROC Curves and compute AUC values ----------------------------------
lwd = 3
plot(roc1, main="ROC curve", col = 'green', lwd = lwd) # Logistic Regression
plot(roc2, add = T, col = 'red', lwd = lwd) # Neural Networks
plot(roc3, add = T, col = 'orange', lwd = lwd) # Support Vector Machine
plot(roc4, add = T, col = 'blue', lwd = lwd) # Random Forest

abline(0, 1, col = 'gray60', lwd = lwd, lty = 3)

# Legend
legend('right', fill = c('green', 'red', 'orange', 'blue'), c('LogReg', 'NN', 'SVM', 'RF'))


# AUC Values

auc(roc(credit[credit.test,]$default.payment.next.month, as.numeric(pred)))
auc(roc(credit[credit.test,]$default.payment.next.month, as.data.frame(pred.test.svm)$`Not default`))
auc(roc(credit[credit.test,]$default.payment.next.month, as.numeric(pred.test.nnet)))
auc(roc(credit[credit.test,]$default.payment.next.month, as.data.frame(pred.test.rf)$`Not default`))




