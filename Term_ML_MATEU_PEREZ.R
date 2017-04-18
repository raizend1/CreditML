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

#get absolute paths to work
path <- "D:/Documents/MIRI/Semestre 3/Machine Learning/TermProject"
#path <-"E:/Documents/Mis Documentos/MIRI/Semestre 3/Machine Learning/TermProject"
dataDir<-glue(path,"/data")
plotDir<-glue(path,"/plots")
codeDir<-glue(path,"/code")
setwd(codeDir)

# read initial data
data.path<-glue(dataDir,"/","default_of_credit_card_clients.csv")
default.credit.card.data<-read.table(data.path,header = TRUE,sep = ";")
str(default.credit.card.data)
# refer to readme to check all the data details. Data are categorical and continous. We will predict 
# default.payment.next.month as a binary yes (1) no (0)
#plot(default.credit.card.data)

# first check N/A values
which(is.na(default.credit.card.data),arr.ind=TRUE) #there are none

# subset of payment history to check some interesting data - maybe
data.sub.payment.history<-default.credit.card.data[,c(7:12)]

# reduce dimensionality - apply PCA
 
# execute svm -  why svm? answer this on the document
require("kernlab")
n.rows <- nrow(default.credit.card.data)
n.length <-ncol(default.credit.card.data)
# execute cross validation - maybe not because of the quantity
# use k-fold CV with k=10
# k <- 10 
# folds <- sample(rep(1:k, length=N), N, replace=FALSE) 

# get just one third for validation, the rest to train
default.credit.card.data.test <- default.credit.card.data[sample(1:n.rows,size = floor(n.rows*0.3),replace = FALSE),]
default.credit.card.train <- default.credit.card.data[which(default.credit.card.data %ni% default.credit.card.data.test),]

#array for the best parameters
c.best <- c()
epsilon.best <- c()
polynomial.degree.best<-c()

#array for computation time
compu.time<- c()




