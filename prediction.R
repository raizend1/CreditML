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

# Initialise workspace, remove old objects for safety resons and define a utility function
rm(list=ls(all=TRUE))
dev.off()
set.seed(123)

#setwd('/Users/cesc/Documents/UPC MIRI-FIB/2Q/Machine Learning/Project/CreditML_Project/code/')
source("Term_ML_MATEU_PEREZ_utility_functions.R")
source("workingDir.R")
setwd(codeDir)

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


# Data Loading and Preprocessing ------------------------------------------

# Read initial data
load('Environment_EDA.RData')

str(credit)
dim(credit)
# [1] 30000    25

# We have a dataset with 30000 rows and 25 variables. All variables are defined as continuous integers,
# and some of them need to be changed to categorical.

# Change 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE' and 'default.payment.next.month' to categorical
factor.indexes <- which(names(credit) %in% c("PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","SEX","EDUCATION","MARRIAGE","default.payment.next.month")) 
credit[,factor.indexes] <- lapply(credit[,factor.indexes], as.factor)

# Remove unnecesary data: ID
credit<- credit[,-1]
factor.indexes<-factor.indexes-1 # update indexes of the factors

# Rename the levels of the categorical values for better unsderstanding
levels(credit$SEX) <- c("Male", "Female")
levels(credit$EDUCATION) <- c("Uk1", "Grad.", "Univ.", "H.School", "Uk2", "Uk3", "Uk4")
levels(credit$MARRIAGE) <- c("Other", "Married", "Single", "Divorced")
levels(credit$default.payment.next.month) <- c("Not default", "Default")
# rename factor variables from columns PAY 6 to 11
for(i in 6:11){
  # levels(credit[,i]) <- c("No consumption", "Paid in full","Use of revolving credit","Payment delay 1M","Payment delay 2M",
  #                         "Payment delay 3M","Payment delay 4M","Payment delay 5M","Payment delay 6M","Payment delay 7M",
  #                         "Payment delay 8M")
  levels(credit[,i]) <- c("NC", "PF","URC","PD1","PD2",
                          "PD3","PD4","PD5","PD6","PD7","PD8")
}

str(credit)
summary(credit)
dim(credit)



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

# Test and Train separation -----------------------------------------------

train.idx <- sample.int(nrow(credit), round(nrow(credit) * 0.8), replace = F)
train <- credit[train.idx,]
test <- credit[-train.idx,]

# 80% Train - 20% Test

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



