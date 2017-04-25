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

# refer to readme to check all the data details. Data are categorical and continous. We will predict 
# default.payment.next.month as a binary yes (1) no (0)

#plot(credit)

#change some data to categorical
tor.nodes[,grepl("Flag...", names(tor.nodes))] <- lapply(tor.nodes[, flags.indexes],as.factor)

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



