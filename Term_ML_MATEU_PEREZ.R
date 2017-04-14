####################################################################################################
# Machine Learning - MIRI Master
# Lecturer: Lluıs A. Belanche, belanche@cs.upc.edu
# Term project
#
# This script represents XXXXXXXXXXXXXXXXXXXXXXXX
#
# Date: 
# Cesc Mateu
# cesc.mateu@gmail.com
# Francisco Pèrez
# pacogppl@gmail.com
#####################################################################################################

#remove old objects for safety resons
rm(list=ls(all=TRUE))
dev.off()

set.seed(123)

#utility function
glue<-function(...){paste(...,sep="")}

#get absolute paths to work
source("workingDir.R")

# read initial data
data.path<-glue(dataDir,"/","default_of_credit_card_clients.csv")
default.credit.card.data<-read.table(data.path,header = TRUE,sep = ";")
str(default.credit.card.data)
# refer to readme to check all the data details. Data are categorical and continous. We will predict 
# default.payment.next.month as a binary yes (1) no (0)

# first check N/A values
which(is.na(default.credit.card.data),arr.ind=TRUE) #there are none

# subset of payment history to check some interesting data - maybe
data.sub.payment.history<-default.credit.card.data[,c(7:12)]

#chechk where are the N/A values
zero.indexes<-unique(which(data2.sub == 0 | data2.sub == -2,arr.ind=TRUE)[,1])
length(zero.indexes)
# [1] 25939
dim(data2[-zero.indexes,])
dim(data2)
# [1] 4061



