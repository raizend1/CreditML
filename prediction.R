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


# Logistic Regression Model -----------------------------------------------

logReg <- glm (, data = credit, family = binomial(link=logit)) 







