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

#remove old objects for safety resons
rm(list=ls(all=TRUE))
dev.off()

set.seed(123)

#utility function
glue<-function(...){paste(...,sep="")}

source("workingDir.R")

setwd(codeDir)

# read initial data
data.path <- glue(dataDir,"/","default_of_credit_card_clients.csv")
credit <- read.table(data.path, header = TRUE,sep = ";")
str(credit)
dim(credit)

# Change 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE' and 'default.payment.next.month' to categorical
factor.cols <- which(names(credit)%in%c("SEX","EDUCATION","MARRIAGE","default.payment.next.month")) 
credit[,factor.cols] <- lapply(credit[,factor.cols],as.factor)

# Rename categorical values for better unsderstanding
levels(credit$SEX) <- c("Male", "Female")
levels(credit$EDUCATION) <- c("Unknown1", "Graduate", "University", "High School", "Unknown2", "Unknown3", "Unknown4")
levels(credit$MARRIAGE) <- c("Other", "Married", "Single", "Divorced")
levels(credit$default.payment.next.month) <- c("Not default", "Default")

str(credit)

<<<<<<< HEAD
# refer to readme to check all the data details. Data are categorical and continous. We will predict 
# default.payment.next.month as a binary yes (1) no (0)
=======
levels(credit.sub$SEX) <- c("Male", "Female")
levels(credit.sub$EDUCATION) <- c("Unknown1", "Graduate", "University", "High School", "Unknown2", "Unknown3", "Unknown4")
levels(credit.sub$MARRIAGE) <- c("Other", "Married", "Single", "Divorced")
levels(credit.sub$default.payment.next.month) <- c("Not default", "Default")
>>>>>>> 521dd2c34ee8a8ada95464c52a0c1da9dbb8109b

#### Exploratory Data Analysis Cesc ####
# Let's work first with just the variables 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE' and 'default.payment.next.month'
library(ggplot2)

str(credit)

### Sex Exploratory Analysis ###
# How many males and females do we have?
ggplot(data = credit.sub, mapping = aes(x = SEX)) + 
  geom_bar()

# We have a lot more females than males in our dataset.

# How many Default's and Not-Defaults's do we have for each sex?
ggplot(data = credit.sub, mapping = aes(x = default.payment.next.month, ..count..)) + 
  geom_bar(mapping = aes(fill = SEX), position = "dodge") 

# It seems that females tend to have less default payments, 
# lets compute the exact proportion to see if there is some kind of bias.
(t <- as.data.frame(with(data = credit.sub, table(SEX, default.payment.next.month))))
t$Freq[t$SEX == "Male" & t$default.payment.next.month == "Default"] / sum(t$Freq[t$SEX == "Male"])
t$Freq[t$SEX == "Female" & t$default.payment.next.month == "Default"] / sum(t$Freq[t$SEX == "Female"])

# As we see, the proportion of males with default payment is 0.241, and the proportion of females is 0.207.
# Indeed, males have a higher probability of default payment.

<<<<<<< HEAD
=======






# refer to readme to check all the data details. Data are categorical and continous. We will predict 
# default.payment.next.month as a binary yes (1) no (0)

#plot(credit)
>>>>>>> 521dd2c34ee8a8ada95464c52a0c1da9dbb8109b

#*******************************************************************************************************
#       initial exploration of education
#*******************************************************************************************************
# a count of all the values to get an initial idea
ggplot(credit, aes(x=EDUCATION)) +
  geom_bar(position="dodge", colour="black") + 
  geom_text(stat='count',aes(label=..count..),vjust=-1)
# we can see that the 4 "unknown" values are very few, comparing them with the others
# also university is most present in this data

# a count check of all the education respect to default payment
ggplot(credit, aes(x=default.payment.next.month, y=EDUCATION)) +
  geom_bar(position="dodge")
  geom_text(stat='count',aes(label=..count..),vjust=-1)

# first check N/A values
which(is.na(credit),arr.ind=TRUE) #there are none

# subset of payment history to check some interesting data - maybe
data.sub.payment.history<-credit[,c(7:12)]

# reduce dimensionality - apply PCA
 
# execute svm -  why svm? answer this on the document
require("kernlab")
n.rows <- nrow(credit)
n.length <-ncol(credit)

# execute cross validation - maybe not because of the quantity
# use k-fold CV with k=10
# k <- 10 
# folds <- sample(rep(1:k, length=N), N, replace=FALSE) 

# get just one third for validation, the rest to train
credit.test <- credit[sample(1:n.rows,size = floor(n.rows*0.3),replace = FALSE),]
credit.train <- subset(credit, !(ID %in% credit.test$ID))

#array for the best parameters
c.best <- c()
epsilon.best <- c()
gamma.best<-c()
polynomial.degree.best<-c()

#array for computation time
compu.time<- c()

# use svm
library(e1071)
model2 <- svm(credit.train[,-25],credit.train[,25],epsilon=0.01,gamma=200, C=100)
lines(credit.train[,-25],predict(model2,credit.train[,-25]),col="green")
credit.svm<-ksvm(credit.train[,-25],credit.train[,25],epsilon=0.01, C=100)



