####################################################################################################
# Machine Learning - MIRI Master
# Lecturer: Lluis A. Belanche, belanche@cs.upc.edu
# Term project
#
# This script represents XXXXXXXXXXXXXXXXXXXXXXXX
#
# Date: 
# Cesc Mateu
# cesc.mateu@gmail.com
# Francisco Perez
# pacogppl@gmail.com
#####################################################################################################


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

# Read initial data
load('Environment_EDA.RData')


# Introduction to the prediction section ----------------------------------

# In this project, we want to predict a binary variable. 'Default.payment.next.month' has two different levels: 
# 'Default' and 'Not Default'. We have 23 different predictors: 14 are continuous and 9 are categorical. 
# We will try different classification models and see which one adapts better to our problem:

# 1. Logistic Regression
# 2. Support Vector Machines (SVM)
# 3. Random Forests
# 4. Neural Networks


# Target variable distribution --------------------------------------------

# Before going into the first type of model, we wanted to know which is our upper error threshold.
# Let's see how much error we would get if we'd always predict the majority class.
table(credit$default.payment.next.month)
# Not default     Default 
# 23167        6533 
6533 / (23167 + 6533) # 0.2199663
# The classifier that always predicts the majority class ('Not default') would give us a 22.12% of error.

# 80% Train - 20% Test

# Then we check if the test and train samples are balanced

# Train and test splitting ------------------------------------------------
rbind(noquote(table(credit$default.payment.next.month)),sapply(prop.table(table(credit$default.payment.next.month))*100, function(u) noquote(sprintf('%.2f%%',u))))
# Not default Default 
# [1,] "23167"     "6533"  
# [2,] "78.00%"    "22.00%"

# As we can see, the distribution of the data is unequal, if we take train and test samples as it is, 
# we could get the model to ignore one of the classes (in this case the Default one, as it is the smallest), 
# so we need to solve this in a simple way, we will take at random the same number of Not Default samples as 
# the Default ones. Once this is done, we can split this subset into train and test data.

set.seed(555)

nd.indexes <- sample(which(credit$default.payment.next.month == "Not default"),6533)
d.indexes <- which(credit$default.payment.next.month == "Default")
subsample.index <- sort(c(nd.indexes,d.indexes))

table(credit[subsample.index,]$default.payment.next.month)

#balanced
credit.train <- sample(subsample.index, size = ceiling(length(subsample.index)*0.8))
credit.test <- subset(subsample.index, !(subsample.index %in% credit.train))

#not balanced
train <- sample(dim(credit)[1], size = ceiling(dim(credit)[1]*0.8))

credit.x <- credit[train,-24]
credit.y <- credit[train,24]

# Initial Logistic Regression Model -----------------------------------------------

# Let's start by fitting a Logistic Regression Model with all the variables:
train <- cbind(credit.x, default = credit.y)
pay <- c(7,8,9,10,11) 
# We have not considered all the categorical PAY_* variables unless PAY_0. They have 7 modalities each
# and they introduce a lot of inestability at the model. 
train <- train[,-pay]

str(train)

( logReg <- glm(default ~ ., data = train, family = binomial(link=logit)) )
summary(logReg)

# Then we try to simplify the model by eliminating the least important variables progressively 
# using the step() algorithm which penalizes models based on the AIC value.

logReg.step <- step(logReg) # Warning: It takes a minute
summary(logReg.step)

# And then refit the model with the optimized model

logReg <- glm (logReg.step$formula, data = train, family = binomial(link = logit))
summary(logReg)

# Call:
#   glm(formula = logReg.step$formula, family = binomial(link = logit), 
#       data = train)
# 
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -2.6111  -0.9141  -0.3588   0.9857   2.4123  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)       -11.632217 305.754207  -0.038 0.969652    
# LIMIT_BAL          -0.241973   0.026870  -9.005  < 2e-16 ***
#   SEXFemale          -0.138609   0.045521  -3.045 0.002327 ** 
#   EDUCATIONGrad.     12.716498 305.753215   0.042 0.966825    
# EDUCATIONUniv.     12.763158 305.753216   0.042 0.966703    
# EDUCATIONH.School  12.678207 305.753220   0.041 0.966925    
# EDUCATIONUk2       11.649119 305.753565   0.038 0.969608    
# EDUCATIONUk3       11.624046 305.753369   0.038 0.969674    
# EDUCATIONUk4       12.586789 305.753715   0.041 0.967163    
# MARRIAGEMarried     2.067299   0.697197   2.965 0.003025 ** 
#   MARRIAGESingle      1.869946   0.697606   2.681 0.007351 ** 
#   MARRIAGEDivorced    2.334919   0.730835   3.195 0.001399 ** 
#   PAY_0PF             0.304912   0.098190   3.105 0.001901 ** 
#   PAY_0URC           -0.142802   0.099296  -1.438 0.150392    
# PAY_0PD1            0.668708   0.107222   6.237 4.47e-10 ***
#   PAY_0PD2            2.131545   0.121462  17.549  < 2e-16 ***
#   PAY_0PD3            2.031520   0.270610   7.507 6.04e-14 ***
#   PAY_0PD4            1.332077   0.459055   2.902 0.003710 ** 
#   PAY_0PD5            0.295067   0.674604   0.437 0.661826    
# PAY_0PD6           12.288107 261.075674   0.047 0.962460    
# PAY_0PD7           12.046611 215.149908   0.056 0.955348    
# PAY_0PD8           -0.203852   0.799915  -0.255 0.798845    
# BILL_AMT3           0.058685   0.011758   4.991 6.00e-07 ***
#   BILL_AMT4           0.038871   0.011737   3.312 0.000927 ***
#   BILL_AMT6           0.053299   0.010322   5.164 2.42e-07 ***
#   PAY_AMT1           -0.063208   0.009000  -7.023 2.17e-12 ***
#   PAY_AMT2           -0.070866   0.010308  -6.874 6.22e-12 ***
#   PAY_AMT3           -0.055917   0.009895  -5.651 1.60e-08 ***
#   PAY_AMT4           -0.043887   0.008350  -5.256 1.47e-07 ***
#   PAY_AMT5           -0.057450   0.010210  -5.627 1.84e-08 ***
#   PAY_AMT6           -0.012366   0.008017  -1.543 0.122951    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 14491  on 10452  degrees of freedom
# Residual deviance: 11866  on 10422  degrees of freedom
# AIC: 11928
# 
# Number of Fisher Scoring iterations: 12

# We observe that the weights assigned to the different variables have different orders of magnitude, 
# which is something not desirable. As we saw during the EDA, maybe applying logarithms to some of the
# variables could help.

# Training error

# Nevertheless, we use the model to do some predictions and compute a training error. Typing 'response'
# in the prediction function will return us the predicted probabilities, it is not 'hard-assigning'
# the observations to a class.

pred <- predict (
  object = logReg,
  type = 'response'
)
# We set a probability threshold 'p' from which we will classify an observation to 'Default' 
# or 'Not Default'.

p <- 0.5
predictions <- NULL
predictions[pred >= p] <- 1
predictions[pred < p] <- 0

# We can compute now the confusion matrix

(tab <- with(credit, table(Truth=train$default,Pred=predictions)))
(error.test <- 100*(1-sum(diag(tab))/nrow(train))) # 29.39952 % of training error


confusionMatrix(predictions, )

# Test error
test_aux <- credit[credit.test,]
test <- credit[credit.test,-24]

# We execute the same process but now using the test data.

pred <- predict (
  object = logReg,
  newdata = test,
  type = 'response',
  na.action = na.pass
)


# We set a probability threshold 'p' from which we will classify an observation to 'Default' 
# or 'Not Default'.

p <- 0.5
predictions <- NULL
predictions[pred >= p] <- 1
predictions[pred < p] <- 0

# We can compute now the confusion matrix for the test data.

( tab <- table(Truth = test_aux$default.payment.next.month, Pred=predictions) )
(error.test <- 100*(1-sum(diag(tab))/nrow(test))) # 29.54 % of error as well.

predictions <- as.factor(predictions)
levels(predictions) <- c('Not default','Default')
confusionMatrix(predictions, test_aux$default.payment.next.month)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction    Not default Default
# Not default        1047     513
# Default             249     804
# 
# Accuracy : 0.7084          
# 95% CI : (0.6905, 0.7258)
# No Information Rate : 0.504           
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.4177          
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Sensitivity : 0.8079          
# Specificity : 0.6105          
# Pos Pred Value : 0.6712          
# Neg Pred Value : 0.7635          
# Prevalence : 0.4960          
# Detection Rate : 0.4007          
# Detection Prevalence : 0.5970          
# Balanced Accuracy : 0.7092          
# 
# 'Positive' Class : Not default 

# Training error is very similar to test error, which means that we are probably 
# not overfitting the dataset with the model.

# Load some libraries to evaluate the binary classifier

library(pROC)
library(caret) 
library(e1071)

# Despite that, the classifier predicts many 'Default' costumers with 'Not default'. Let's compute
# the precision and the accuracy of the model. 

levels <- c('Not default', 'Default')
truth <- test_aux$default.payment.next.month
predictions <- as.factor(predictions)
levels(predictions) <- c('Not default', 'Default')

confusionMatrix(table(truth, predictions))

# Plot the ROC curve
test_aux$prob <- pred
g <- roc(default.payment.next.month ~ prob, data = test_aux)
plot(g)
auc(g)

# Area under the curve: 0.7661

# We saw during the EDA that some of the variables had very skewed distributions. The application
# of logarithms could help improve our prediction.

# Support Vector Machine with Radial Basis Function Kernel ------------------------------------------

require(kernlab)
require(caret)

# We are using the train function from caret for a Kernel RBF SVM Machine, using a 10 fold cv repeted five times,
# To have more information on how well this method works with this configuration, we want to be able to have a 
# look at the folds, and for each of them how close the predicted values were to the actual values, so we set the 
# savePred parameter as true.

set.seed(555)
svm.train.control<- trainControl(method = "repeatedcv",number = 10,repeats = 5,savePred=TRUE)
#this train will take a whille, as cv is computational costly
cl <- makeCluster(detectCores())
registerDoParallel(cl)
start.time<-proc.time()
svm.tune <- train(x=credit.x[,-factor.indexes], y= credit.y,
                  method = "svmRadial", trControl = svm.train.control)
end.time<-proc.time()
time.svm<- end.time-start.time
stopCluster(cl)

# SVM First round results
time.svm
# user  system elapsed 
# 21.72    1.36 1272.49 # took about 21 minutes

head(svm.tune$pred)
#      pred         obs      rowIndex      sigma    C    Resample
# 1 Not default Not default       11    0.09403567 0.25 Fold01.Rep1
# 2 Not default Not default       17    0.09403567 0.25 Fold01.Rep1
# 3     Default Not default       20    0.09403567 0.25 Fold01.Rep1
# 4     Default     Default       30    0.09403567 0.25 Fold01.Rep1
# 5     Default Not default       87    0.09403567 0.25 Fold01.Rep1
# 6 Not default     Default       94    0.09403567 0.25 Fold01.Rep1
svm.tune
# Resampling results across tuning parameters:
#   
#   C     Accuracy   Kappa    
# 0.25  0.6582040  0.3182430
# 0.50  0.6572337  0.3166111
# 1.00  0.6585812  0.3193315
# 
# Tuning parameter 'sigma' was held constant at a value
# of 0.09374442
# Accuracy was used to select the optimal model using 
# the largest value.
# The final values used for the model were sigma =
#   0.09374442 and C = 1.

# Results with kernlab after tunning --------------------------------------
credit.svm <- ksvm(default.payment.next.month ~.,
                   data=credit[credit.train,], kernel='rbfdot', type = "C-svc",
                   kpar=list(sigma=0.09374442),C=1,prob.model = TRUE)

sparsity <-1 - (credit.svm@nSV / dim(credit)[1])
# [1] 0.7537037

pred.test.svm <- predict(credit.svm, newdata=credit[credit.test,])

confusionMatrix(pred.test.svm, credit[credit.test,]$default.payment.next.month)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    Not default Default
# Not default        1024     517
# Default             272     800
# 
# Accuracy : 0.698         
# 95% CI : (0.68, 0.7156)
# No Information Rate : 0.504         
# P-Value [Acc > NIR] : < 2.2e-16     
# 
# Kappa : 0.397         
# Mcnemar's Test P-Value : < 2.2e-16     
#                                         
#             Sensitivity : 0.7901        
#             Specificity : 0.6074        
#          Pos Pred Value : 0.6645        
#          Neg Pred Value : 0.7463        
#              Prevalence : 0.4960        
#          Detection Rate : 0.3919        
#    Detection Prevalence : 0.5897        
#       Balanced Accuracy : 0.6988        
#                                         
#        'Positive' Class : Not default

pred.test.svm <- predict(credit.svm, newdata=credit[credit.test,],type="prob")
invisible(require(ROCR))
pred <- prediction(as.data.frame(pred.test.svm)$`Not default`, credit$default.payment.next.month[credit.test])
roc <- performance(pred,measure="tpr",x.measure="fpr")
plot(roc, main="ROC curve")
abline(0,1,col="blue")
library(pROC)
auc(roc(credit[credit.test,]$default.payment.next.month, as.data.frame(pred.test.svm)$`Not default`))
# Area under the curve: 0.7557

# Neural Networks ------------------------------------------

library(nnet)
library(devtools)

# Function to plot the neural networks
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')

# Train: (credit.x, credit.y)
# Test: credit[credit.test,-24]

train <- cbind(credit.x, default = credit.y)
test <- credit[credit.test,]

dim(train)
# [1] 10092    24
dim(test)
# [1] 2522   24


# Training the neural network. One output layer, one hidden layer with 20 neurons.
model.nnet <- nnet( default ~ . ,
                   data = train,
                   size=20,
                   MaxNWts = 2000,
                   maxit=500)

# Output
model.nnet 
# Plot the neural network
plot.nnet(model.nnet)
# Fitting criterion (aka error function)
model.nnet$value
# Fitted values for the training data
model.nnet$fitted.values
# Residuals
model.nnet$residuals
# Weights
model.nnet$wts

# Training error. Accuracy: 0.7153
pred.train <- as.factor(predict (model.nnet, type="class"))
confusionMatrix(pred.train, train[,24])

# Test error. Accuracy: 0.6987
pred.test <- as.factor(predict(model.nnet, newdata = test[,-24], type = 'class'))
confusionMatrix(pred.test, test[,24])

# As you can see, some weights are large (two orders of magnitude larger then others)
# This is no good, since it makes the model unstable (ie, small changes in some inputs may
# entail significant changes in the network, because of the large weights)
# One way to avoid this is by regularizing (decay parameter).

model.nnet <- nnet( default ~ . ,
                    data = train,
                    size=20,
                    MaxNWts = 2000,
                    maxit=500,
                    decay = 0.5 )

summary(model.nnet) # Now the weights are more similar, and some of them have been converted to 0.

# Training error. Accuracy: 0.7644
pred.train <- as.factor(predict (model.nnet, type="class"))
confusionMatrix(pred.train, train[,24])

# Test error. Accuracy: 0.6959
pred.test <- as.factor(predict(model.nnet, newdata = test[,-24], type = 'class'))
confusionMatrix(pred.test, test[,24])

pred.test.raw <- predict(model.nnet, newdata = test[,-24], type = 'raw')

pred.test[1:10]
pred.test.raw[1:10]

names(test)

# Plot the ROC curve
test$prob <- pred.test.raw
g <- roc(default.payment.next.month ~ prob, data = test)
plot(g)
auc(g)
# Area under the curve: 0.7541

# Adjustment of the parameters

# For a specific model, in our case the neural network, the function train() in {caret} 
# uses a "grid" of model parameters and trains using a given resampling method (in our case we 
# will be using 10x10 CV). All combinations are evaluated, and the best one (according to 10x10 CV) 
# is chosen and used to construct a final model, which is refit using the whole training set

(decays <- 10^seq(-3,0,by=0.2))
trc <- trainControl (method="repeatedcv", number=10, repeats=5)

start.time <- proc.time()

## WARNING: This may take a long time
model.10x10CV <- train ( default ~., 
                        data = train,
                        method='nnet', 
                        maxit = 300, 
                        trace = FALSE,
                        tuneGrid = expand.grid(.size=10,.decay=decays), 
                        trControl=trc)
end.time <- proc.time()
(time.nn <- end.time - start.time)
#     user   system  elapsed 
#   3211.993   24.032 3273.546 

model.10x10CV

# Neural Network
# 10453 samples
# 23 predictor
# 2 classes: 'Not default', 'Default'
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 5 times)
# Summary of sample sizes: 9407, 9408, 9408, 9408, 9408, 9408, ...
# Resampling results across tuning parameters:
# 
#   decay        Accuracy   Kappa
# 0.001000000  0.7005649  0.4007446
# 0.001584893  0.7019033  0.4034506
# 0.002511886  0.6944432  0.3884516
# 0.003981072  0.7009464  0.4015059
# 0.006309573  0.7010417  0.4016807
# 0.010000000  0.7020928  0.4037975
# 0.015848932  0.7041991  0.4080645
# 0.025118864  0.6966407  0.3929625
# 0.039810717  0.7011384  0.4019282
# 0.063095734  0.6999909  0.3996089
# 0.100000000  0.7002759  0.4002038
# 0.158489319  0.6956855  0.3910378
# 0.251188643  0.6968334  0.3933196
# 0.398107171  0.6986499  0.3969924
# 0.630957344  0.7014244  0.4025097
# 1.000000000  0.7010412  0.4017222
# 
# Tuning parameter 'size' was held constant at a value of 10
# Accuracy was used to select the optimal model using  the largest value.
# The final values used for the model were size = 10 and decay = 0.01584893.

# Training error
pred.train <- predict (model.10x10CV, type="prob")

# Transforming the probabilities into classes
predictions.train <- factor()
levels(predictions.train) <- c('Default', 'Not default')
for (i in 1:nrow(pred.train)){
  if(pred.train[i,1] >= 0.5){
    predictions.train[i] <- 'Not default' 
  }else{
    predictions.train[i] <- 'Default'
  }
}
confusionMatrix(predictions.train, train[,24])

# Confusion Matrix and Statistics
# 
# Reference
# Prediction    Not default Default
# Not default        3797    1727
# Default            1069    3499
# 
# Accuracy : 0.7229          
# 95% CI : (0.7141, 0.7317)
# No Information Rate : 0.5178          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.4478          
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Sensitivity : 0.7803          
# Specificity : 0.6695          
# Pos Pred Value : 0.6874          
# Neg Pred Value : 0.7660          
# Prevalence : 0.4822          
# Detection Rate : 0.3762          
# Detection Prevalence : 0.5474          
# Balanced Accuracy : 0.7249          
# 
# 'Positive' Class : Not default 

# Test error.
pred.test <- predict(model.10x10CV, newdata = test[,-24], type = 'prob')

# Transforming the probabilities into classes
predictions.test <- factor()
levels(predictions.test) <- c('Default', 'Not default')
for (i in 1:nrow(pred.test)){
  if(pred.train[i,1] >= 0.5){
    predictions.test[i] <- 'Not default' 
  }else{
    predictions.test[i] <- 'Default'
  }
}

pred.test

confusionMatrix(predictions.test, test[,24])

# Cesc: I think I have overfitted the model

# Confusion Matrix and Statistics
# 
# Reference
# Prediction    Not default Default
# Not default         636     726
# Default             579     581
# 
# Accuracy : 0.4826          
# 95% CI : (0.4629, 0.5023)
# No Information Rate : 0.5182          
# P-Value [Acc > NIR] : 0.9998          
# 
# Kappa : -0.0319         
# Mcnemar's Test P-Value : 5.31e-05        
# 
# Sensitivity : 0.5235          
# Specificity : 0.4445          
# Pos Pred Value : 0.4670          
# Neg Pred Value : 0.5009          
# Prevalence : 0.4818          
# Detection Rate : 0.2522          
# Detection Prevalence : 0.5400          
# Balanced Accuracy : 0.4840          
# 
# 'Positive' Class : Not default  


# Plot the ROC curve
test_aux$prob <- pred.test
g <- roc(default.payment.next.month ~ prob, data = test_aux)
plot(g)


# Random Forests ----------------------------------------------------------
library(randomForest)

# First as before, we use the train function to estimate the values for random forest
# First scenario, using oob method
rf.train.control<- trainControl(method = "oob",savePred=TRUE)

cl <- makeCluster(detectCores())
registerDoParallel(cl)
start.time<-proc.time()
rf.tune <- train(x=credit.x[,-factor.indexes], y= credit.y,
                  method = "rf", trControl = rf.train.control, metric = "Accuracy")
end.time<-proc.time()
time.rf<- end.time-start.time
stopCluster(cl)

#bestmtry <- tuneRF(x=credit.x[,-factor.indexes], y= credit.y, stepFactor=1.5, improve=1e-5, ntree=500)

time.rf
# user  system elapsed 
# 43.37    0.86  177.72

rf.tune
# Random Forest 
# 
# 23760 samples
# 14 predictor
# 2 classes: 'Not default', 'Default' 
# 
# No pre-processing
# Resampling results across tuning parameters:
#   
#   mtry  Accuracy   Kappa    
# 2    0.7917088  0.1902829
# 8    0.7873737  0.2121606
# 14    0.7852694  0.2179754
# 
# Accuracy was used to select the optimal model using  the largest value.
# The final value used for the model was mtry = 2.

pred.test.rf <- predict(rf.tune, credit[credit.test,],type="raw")
confusionMatrix(pred.test.rf, credit[credit.test,]$default.payment.next.month)

# Confusion Matrix and Statistics - with oob
# 
# Reference
# Prediction    Not default Default
# Not default        1287     270
# Default               9    1047
# 
# Accuracy : 0.8932          
# 95% CI : (0.8808, 0.9048)
# No Information Rate : 0.504           
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.7868          
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Sensitivity : 0.9931          
# Specificity : 0.7950          
# Pos Pred Value : 0.8266          
# Neg Pred Value : 0.9915          
# Prevalence : 0.4960          
# Detection Rate : 0.4925          
# Detection Prevalence : 0.5959          
# Balanced Accuracy : 0.8940          
# 
# 'Positive' Class : Not default 


# Second scenario, using cv method
rf.train.control<- trainControl(method = "repeatedcv",number = 10,repeats = 5,savePred=TRUE)

cl <- makeCluster(detectCores())
registerDoParallel(cl)
start.time<-proc.time()
rf.tune <- train(x=credit.x[,-factor.indexes], y= credit.y,
                 method = "rf", trControl = rf.train.control)
end.time<-proc.time()
time.rf<- end.time-start.time
stopCluster(cl)

time.rf 
# user  system elapsed 
# 34.92    2.51 3740.32

head(rf.tune$pred)

rf.tune
# Random Forest 
# 
# 23760 samples
# 14 predictor
# 2 classes: 'Not default', 'Default' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 5 times) 
# Summary of sample sizes: 21383, 21384, 21384, 21384, 21384, 21384, ... 
# Resampling results across tuning parameters:
#   
#   mtry  Accuracy   Kappa    
# 2    0.7917340  0.1856724
# 8    0.7884933  0.2143070
# 14    0.7857155  0.2154263
# 
# Accuracy was used to select the optimal model using  the largest value.
# The final value used for the model was mtry = 2.

pred.test.rf <- predict(rf.tune, credit[credit.test,],type="raw")
confusionMatrix(pred.test.rf, credit[credit.test,]$default.payment.next.month)

# Confusion Matrix and Statistics with cv 10 fold 5 times
# 
# Reference
# Prediction    Not default Default
# Not default        1286     266
# Default              10    1051
# 
# Accuracy : 0.8944         
# 95% CI : (0.882, 0.9059)
# No Information Rate : 0.504          
# P-Value [Acc > NIR] : < 2.2e-16      
# 
# Kappa : 0.7891         
# Mcnemar's Test P-Value : < 2.2e-16      
#                                          
#             Sensitivity : 0.9923         
#             Specificity : 0.7980         
#          Pos Pred Value : 0.8286         
#          Neg Pred Value : 0.9906         
#              Prevalence : 0.4960         
#          Detection Rate : 0.4922         
#    Detection Prevalence : 0.5940         
#       Balanced Accuracy : 0.8952         
#                                          
#        'Positive' Class : Not default

#So as we cn see, the values are pretty similar, in this case we prefer the OOB error, because it is not
# as computaional expensive as CV and offers good results

pred.test.rf <- predict(rf.tune, credit[credit.test,],type="prob")
invisible(require(ROCR))
pred <- prediction(as.data.frame(pred.test.rf)$`Not default`, credit$default.payment.next.month[credit.test])
roc <- performance(pred,measure="tpr",x.measure="fpr")
plot(roc, main="ROC curve")
abline(0,1,col="blue")
library(pROC)
auc(roc(credit[credit.test,]$default.payment.next.month, as.data.frame(pred.test.rf)$`Not default`))
# Area under the curve: 0.9718
