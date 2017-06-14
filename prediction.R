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

# Read initial data
load('Environment_EDA.RData')

# Introduction to the prediction section ----------------------------------

# In this project, we want to predict a binary variable. 'Default.payment.next.month' has two different levels: 
# 'Default' and 'Not Default'. We have 23 different predictors: 14 are continuous and 9 are categorical. 
# We will try different classification models and see which one adapts better to our problem:

# 1. Logistic Regression
# 2. Linear/Quadratic Discriminant Analysis (LDA/QDA)
# 3. Support Vector Machines (SVM)
# 4. Random Forests
# 5. Neural Networks

# Before going into the first type of model, we wanted to know which is our upper error threshold.
# Let's see how much error we would get if we'd always predict the majority class.
table(credit$default.payment.next.month)
# Not default     Default 
# 23364        6636 
6636 / (23364 + 6636) # 0.2212
# The classifier that always predicts the majority class ('Not default') would give us a 22.12% of error.
# This is our threshold. Any model with a higher error than this can be automatically discarded.

# 80% Train - 20% Test

#*****************************************************************************************#
#                                        4. Modeling                                      #
#*****************************************************************************************#

# We first check if the test and train samples are balanced
rbind(noquote(table(credit$default.payment.next.month)),sapply(prop.table(table(credit$default.payment.next.month))*100, function(u) noquote(sprintf('%.2f%%',u))))
# Not default Default 
# [1,] "23167"     "6533"  
# [2,] "78.00%"    "22.00%"

# As we can see, the distribution of the data is unequal, if we take train and test samples as it is, 
# we could get the model to ignore one of the classes (in this case the Default one, as it is the smallest), 
# so we need to solve this in a simple way, we will take at random the same number of Not Default samples as 
# the Default ones. Once this is done, we can split this subset into train and test data.

set.seed(555)
nd.indexes<-sample(which(credit$default.payment.next.month == "Not default"),6081)
d.indexes<-which(credit$default.payment.next.month == "Default")
subsample.index<-sort(c(nd.indexes,d.indexes))

table(credit[subsample.index,]$default.payment.next.month)

credit.train <- sample(subsample.index, size = ceiling(length(subsample.index)*0.8))
credit.test <- subset(subsample.index, !(subsample.index %in% credit.train))


#*****************************************************************************************#
#                                        SVM Definition                                   #
#*****************************************************************************************#
require(kernlab)
require(caret)

credit.x <- credit.continuos[credit.train,-24]
credit.y <- credit[credit.train,24]

# We are using the train function from caret for a Kernel RBF SVM Machine, using a 10 fold cv repeted five times,
# To have more information on how well this method works with this configuration, we want to be able to have a 
# look at the folds, and for each of them how close the predicted values were to the actual values, so we set the 
# savePred parameter as true.

start.time<-proc.time()
svm.train.control<- trainControl(method = "repeatedcv",number = 10,repeats = 5,savePred=TRUE)
svm.tune <- train(x=credit.x, y= credit.y, 
                  method = "svmRadial", trControl = svm.train.control)
end.time<-proc.time()
time.svm<- end.time-start.time

# SVM First round results
time.svm
# user  system elapsed 
# 1.52    0.09    1.67
head(svm.tune$pred)
#      pred         obs      rowIndex      sigma    C    Resample
# 1 Not default Not default       11    0.09403567 0.25 Fold01.Rep1
# 2 Not default Not default       17    0.09403567 0.25 Fold01.Rep1
# 3     Default Not default       20    0.09403567 0.25 Fold01.Rep1
# 4     Default     Default       30    0.09403567 0.25 Fold01.Rep1
# 5     Default Not default       87    0.09403567 0.25 Fold01.Rep1
# 6 Not default     Default       94    0.09403567 0.25 Fold01.Rep1
svm.tune
# Tuning parameter 'sigma' was held constant at a value of 0.09403567
# Accuracy was used to select the optimal model using  the largest value.
# The final values used for the model were sigma = 0.09403567 and C = 0.25.

# Now we can proceed with the second round by refined the parameter space
# using the best values of sigma and C
grid <- expand.grid(sigma = c(0.075,0.08, 0.095, 0.1,0.15),
                    C = c(0.05, 0.15, 0.25, 0.35, 0.4))

start.time<-proc.time()
svm.tune <- train(x=credit.x[credit.train,], y= credit.y[credit.train], 
                  method = "svmRadial", tuneGrid = grid)
end.time<-proc.time()
time.svm<- end.time-start.time


nnet.tune<-train(credit.x[credit.train,], credit.y[credit.train],
                 method = "nnet",
                 preProcess = c("pca"),
                 trControl= trainControl(method = "cv", number = 10),
                 tuneGrid = expand.grid(.size=c(1,5,10, 15, 20, 25, 30, 35, 40),.decay=c(0,0.001,0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4)))# Check importance of variables






# Initial Logistic Regression Model -----------------------------------------------

# Let's start by fitting a Logistic Regression Model with all the variables:

( logReg <- glm (default.payment.next.month ~ ., data = train, family = binomial(link=logit)) )
summary(logReg)

# Then we try to simplify the model by eliminating the least important variables progressively 
# using the step() algorithm which penalizes models based on the AIC value.

# logReg.step <- step(logReg) # Warning: It takes a while
summary(logReg.step)

# And then refit the model with the optimized model

logReg <- glm (logReg.step$formula, data = train, family = binomial(link = logit))
summary(logReg)

# We observe that the weights assigned to the different variables have different orders of magnitude, 
# which is something not desirable. As we saw during the EDA, maybe applying logarithms to some of the
# variables could help.

# Training error

# Nevertheless, we use the model to do some predictions and compute a training error. Typing 'response'
# in the prediction function will return us the predicted probabilities, it is not 'hard-assigning'
# the observations to a class.

pred <- predict (
  object = logReg,
  newdata = train,
  type = 'response'
                )
# We set a probability threshold 'p' from which we will classify an observation to 'Default' 
# or 'Not Default'.

p <- 0.5
predictions <- NULL
predictions[pred >= p] <- 1
predictions[pred < p] <- 0

# We can compute now the confusion matrix

(tab <- with(credit, table(Truth=default.payment.next.month[train.idx],Pred=predictions)))
(error.test <- 100*(1-sum(diag(tab))/nrow(train))) # ~17 % of training error

# Test error

# We execute the same process but now using the test data.

pred <- predict (
  object = logReg,
  newdata = test,
  type = 'response'
)

# We set a probability threshold 'p' from which we will classify an observation to 'Default' 
# or 'Not Default'.

p <- 0.5
predictions <- NULL
predictions[pred >= p] <- 1
predictions[pred < p] <- 0

# We can compute now the confusion matrix for the test data.

(tab <- with(test, table(Truth=default.payment.next.month,Pred=predictions)))
(error.test <- 100*(1-sum(diag(tab))/nrow(test))) # ~ 17 % of error as well.

# Training error is very similar to test error, which means that we are probably 
# not overfitting the dataset with the model. It is lower from the threshold that we initially
# marked, 22%.

# Load some libraries to evaluate the binary classifier

library(pROC)
library(caret) 
library(e1071)

# Despite that, the classifier predicts many 'Default' costumers with 'Not default'. Let's compute
# the precision and the accuracy of the model. 

levels <- c('Not default', 'Default')
truth <- test$default.payment.next.month
predictions <- as.factor(predictions)
levels(predictions) <- c('Not default', 'Default')

confusionMatrix(table(truth, predictions))

# Plot the ROC curve
credit$prob <- pred
g <- roc(default.payment.next.month ~ prob, data = credit)
plot(g)

# Not very good results...

# We saw during the EDA that some of the variables had very skewed distributions. The application
# of logarithms could help improve our prediction.


# Logistic Regression Model v2 ---------------------------------


# Support Vector Machine with Radial Basis Function Kernel ------------------------------------------

# We will use the very powerful 'train' function to train a SVM with a Radial Basis Function acting
# as a Kernel function. First we will adapt our data to the arguments of the function.

library(kernlab)
library(caret)
library(tidyverse)

svmModelInfo <- getModelInfo(model = "svmRadial", regex = FALSE)[[1]]
names(svmModelInfo)

# As a first training of the SVM model, we won't take into account the variables PAY_*. The variables
# BILL_ATM* and PAY_ATM* have been gathered all together into a single variable, BILL_ATM_TOTAL and
# PAY_ATM_TOTAL, as we have demonstrated through the corelation matrix and the PCA that they are very
# related. To gather them we have just applied an 'average' function for each individual.

# Matrix of predictors with the new defined variables

train$BILL_ATM_TOTAL <- with(train, (BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6) / 6)
train$PAY_ATM_TOTAL <- with(train, (PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6) / 6)


predictors_SVM <- train[c(1,5,25,26)]
target_SVM <- train[,24]

summary(predictors_SVM)
# train %>% mutate(
#   BILL_ATM_TOTAL = (BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6) / 6 ,
#   PAY_ATM_TOTAL = (PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6) / 6
# ) %>% 
# select(LIMIT_BAL, AGE, BILL_ATM_TOTAL, PAY_ATM_TOTAL)


# Training of the SVM with basis radial function
train(
  x = predictors_SVM,
  y = target_SVM,
  method = 'svmRadial',
  metric = 'Accuracy',
  maximize = TRUE
)


# Neural Networks ------------------------------------------

library(nnet)

model.nnet <- nnet( ~.,
                   data = Admis, 
                   subset=learn, 
                   size=20,
                   maxit=200)
## Let's start by standardizing the inputs; this is important to avoid network stagnation (premature convergence)



## For a specific model, in our case the neural network, the function train() in {caret} 
# uses a "grid" of model parameters and trains using a given resampling method (in our case we 
# will be using 10x10 CV). All combinations are evaluated, and the best one (according to 10x10 CV) 
# is chosen and used to construct a final model, which is refit using the whole training set

## Thus train() returns the constructed model (exactly as a direct call to nnet() would)

## In order to find the best network architecture, we are going to explore two methods:

## a) Explore different numbers of hidden units in one hidden layer, with no regularization
## b) Fix a large number of hidden units in one hidden layer, and explore different regularization values (recommended)

## doing both (explore different numbers of hidden units AND regularization values)
# is usually a waste of computing resources (but notice that train() would admit it)

nnet_model <- train (admit ~.,
                        data = train,
                        method='nnet', 
                        maxit = 500, 
                        trace = FALSE,
                        tuneGrid = expand.grid(.size=sizes,.decay=0)
                     )



################ Neural Network plotting function by fadwa123 ##################
library(devtools)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')
################################################################################





# Random Forests ----------------------------------------------------------



