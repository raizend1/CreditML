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
factor.indexes <- which(names(credit)%in%c("PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","SEX","EDUCATION","MARRIAGE","default.payment.next.month")) 
credit[,factor.indexes] <- lapply(credit[,factor.indexes],as.factor)

# Rename categorical values for better unsderstanding
levels(credit$SEX) <- c("Male", "Female")
levels(credit$EDUCATION) <- c("Unknown1", "Graduate", "University", "High School", "Unknown2", "Unknown3", "Unknown4")
levels(credit$MARRIAGE) <- c("Other", "Married", "Single", "Divorced")
levels(credit$default.payment.next.month) <- c("Not default", "Default")

str(credit)
summary(credit)

#***************************************************************************#
#               Initial Exploratory analysis                                #
#***************************************************************************#
# remove unnecesary data: ID
credit<- credit[,-1]
factor.indexes<-factor.indexes-1 # update indexes of the factors

# Are there any zero variance predictors?   
library("caret")
x = nearZeroVar(credit, saveMetrics = TRUE)
str(x)
x[x[,"zeroVar"] > 0, ] 
x[x[,"zeroVar"] + x[,"nzv"] > 0, ] 
#there are none, we can conclude that all the predictors are relevant for the moment

# First check N/A values
which(is.na(credit),arr.ind=TRUE) #there are none

# check distribution of data
draw.plot<-function(input.data,type){
  #require(ggplot2)
  #require(gridExtra)
  l.data<-length(input.data)
  rounded<-round(sqrt(l.data),0)
  par(mar=c(3,3,2,2))
  par(mfrow=c(rounded-1, rounded+1))
  # for(i in 1:l.data){
  out.plot<-array(dim = l.data)
  #   eval(parse(text=glue(type,"(input.data[,i],main = names(input.data)[i])")))}
  switch(type,
         histogram={for(i in 1:l.data){hist(input.data[,i],main = names(input.data)[i],prob=TRUE);lines(density(input.data[,i]),col="blue", lwd=2)}},
         # histogram={out.plot <- lapply(1:14, function(i) ggplot(data=input.data, aes(input.data[,i])) +
         #                                 geom_histogram(aes(y =..density..),breaks=seq(20, 50, by = 2),col="red",fill="green",alpha = .2) +
         #                                 geom_density(col=i) +labs(title=names(input.data)[i],x=element_blank()))},
         plot={for(i in 1:l.data){plot(input.data[,i],main = names(input.data)[i])}},
         stop("Valid plot types are 'histogram', 'plot'"))
  #marrangeGrob(input.data, nrow=rounded, ncol=rounded)
}

#most of the data is not normal, have some very high skewed values, also the scales are radicall different

draw.plot(credit[,-factor.indexes],"histogram")

ggplot(data = credit, mapping = aes(x = AGE, ..count..)) + 
  geom_bar(mapping = aes(fill = AGE), position = "dodge") 

ggplot(data = credit, mapping = aes(x = credit[,-factor.indexes],..count..)) + 
  geom_bar()

#Normalize the data

# subset of payment history to check some interesting data - maybe
data.sub.payment.history<-credit[,c(7:12)]

#### Exploratory Data Analysis Cesc ####
# Let's work first with just the variables 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE' and 'default.payment.next.month'
library(ggplot2)

str(credit)

### Sex Exploratory Analysis ###
# How many males and females do we have?
ggplot(data = credit, mapping = aes(x = SEX)) + 
  geom_bar()

# We have a lot more females than males in our dataset.

# How many Default's and Not-Defaults's do we have for each sex?
ggplot(data = credit, mapping = aes(x = default.payment.next.month, ..count..)) + 
  geom_bar(mapping = aes(fill = SEX), position = "dodge") 

# It seems that females tend to have less default payments, 
# lets compute the exact proportion to see if there is some kind of bias.
(t <- as.data.frame(with(data = credit, table(SEX, default.payment.next.month))))
t$Freq[t$SEX == "Male" & t$default.payment.next.month == "Default"] / sum(t$Freq[t$SEX == "Male"])
t$Freq[t$SEX == "Female" & t$default.payment.next.month == "Default"] / sum(t$Freq[t$SEX == "Female"])

# As we see, the proportion of males with default payment is 0.241, and the proportion of females is 0.207.
# Indeed, males have a higher probability of default payment.


#*****************************************************************************************#
#               initial exploration of education                                          #
#*****************************************************************************************#
# a count of all the values to get an initial idea
ggplot(credit, aes(x=EDUCATION)) +
  geom_bar(position="dodge", colour="black") + 
  geom_text(stat='count',aes(label=..count..),vjust=-1)
# we can see that the 4 "unknown" values are very few, comparing them with the others
# also university is most present in this data

# a count check of all the education respect to default payment
ggplot(credit, aes(x=default.payment.next.month)) +
  geom_bar(mapping = aes(fill = EDUCATION),position="dodge")
  geom_text(stat='count',aes(label=..count..),vjust=-1)

#*****************************************************************************************#
#                              Initial model assumptions                                  #
#*****************************************************************************************#
#check correlation
cor(credit$EDUCATION,credit$default.payment.next.month)

factor.indexes <- head(factor.indexes, -1)
m1<-lm(default.payment.next.month~.,data=credit[,-factor.indexes])
summary(m1)

par(mfrow=c(2,2))
plot(m1)

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



