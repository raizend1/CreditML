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
data.path <- glue(dataDir,"/","default_of_credit_card_clients.csv")
credit <- read.table(data.path, header = TRUE,sep = ";")
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



# Test and Train separation -----------------------------------------------

train.idx <- sample.int(nrow(credit), round(nrow(credit) * 0.8), replace = F)
train <- credit[train.idx,]
test <- credit[-train.idx,]

# 80% Train - 20% Test

# Logistic Regression Model -----------------------------------------------

# Let's start by fitting a Logistic Regression Model with all the variables:

( logReg <- glm (default.payment.next.month ~ ., data = train, family = binomial(link=logit)) )

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
(error.test <- 100*(1-sum(diag(tab))/nrow(train))) # 17.76 % of training error

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

# We can compute now the confusion matrix
(tab <- with(train, table(Truth=default.payment.next.month,Pred=predictions)))
(error.test <- 100*(1-sum(diag(tab))/nrow(test))) # 



