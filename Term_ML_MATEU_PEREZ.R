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


#***************************************************************************#
#                           0. Initialization                               #
#***************************************************************************#

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

#***************************************************************************#
#                    1. Data Loading and Preprocessing                      #
#***************************************************************************#

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

# Rename the levels of the categorical values for better unsderstanding
levels(credit$SEX) <- c("Male", "Female")
levels(credit$EDUCATION) <- c("Unknown1", "Graduate", "University", "High School", "Unknown2", "Unknown3", "Unknown4")
levels(credit$MARRIAGE) <- c("Other", "Married", "Single", "Divorced")
levels(credit$default.payment.next.month) <- c("Not default", "Default")

str(credit)
summary(credit)

# Remove unnecesary data: ID
credit<- credit[,-1]
factor.indexes<-factor.indexes-1 # update indexes of the factors

#***************************************************************************#
#                2. Initial Exploratory Data Analysis (EDA)                 #
#***************************************************************************#

#***************************************************************************#
#                      2.1 Check Zero variance predictors                   #
#***************************************************************************#

# Are there any zero variance predictors? nearZeroVar() diagnoses predictors that have one unique value 
# (i.e. are zero variance predictors) or predictors that are have both of the following characteristics: 
# they have very few unique values relative to the number of samples and the ratio of the frequency of the 
# most common value to the frequency of the second most common value is large.  
library("caret")
x = nearZeroVar(credit, saveMetrics = TRUE)
str(x)
x[x[,"zeroVar"] > 0, ] 
x[x[,"zeroVar"] + x[,"nzv"] > 0, ] 
#There are none, we can conclude that all the predictors are relevant for the moment.

#***************************************************************************#
#                              2.2 Check N/A Values                         #
#***************************************************************************#

# First check N/A values
which(is.na(credit),arr.ind=TRUE) 
md.pattern(credit)
# It can't find any value expressed as 'NA', but there are some rows where all the values for the billing
# statements and the previous payment are 0, this could be treated as a missing value, because if there is 
# a credit card issued, there must be values for this columns, so we treat them as missing values. First 
# a check of how many of this occurrences exist is needed
check.zero.rows<-function(input.data){
  indexes<-NULL
  j<-1
  for(i in 1:dim(input.data)[1]){
    if(all(input.data[i,c(1:dim(input.data)[2])]==0)){
      indexes[j]<- i
      j<-j+1
    }
  }
  return(indexes)
}

num.zeros.index<-check.zero.rows(credit[,c(12:23)])
length(num.zeros.index)
(length(num.zeros.index)*100)/dim(credit)[1]
# so in total there are 795 of this kind of data, it represents 2.65% of the data, but we will discard them anyway
# because this data can produce errors in further steps like the change in scales
credit<-credit[-num.zeros.index,]

# update continuous and factor data
credit.continuos<-credit[,-factor.indexes]
credit.factors<-credit[,factor.indexes]


#######*****************NOTA PARA CESC: talves la parte siguiente esta ya obsoleta por el gr'afico de arriba

################# Analysis of the continuous variables ###################
# define continuous and factor data
credit.continuos<-credit[,-factor.indexes]
credit.factors<-credit[,factor.indexes]

summary(credit.continuos)

###### LIMIT_BAL ######
# Everything seems correct, we have an extreme outlier of an individual with a credit limit of 1.000.000, but it is
# not impossible, just a lucky rich person :).
credit[credit$LIMIT_BAL > 900000,]
initial.histogram(credit,LIMIT_BAL,FALSE)
initial.boxplot(credit,LIMIT_BAL,FALSE)

###### AGE ######
# Nothing weird. The mean age of the individuals is 35.49 years.
initial.histogram(credit,AGE,FALSE)
initial.boxplot(credit,AGE,FALSE)

###### BILL_AMT(X) ######
# We see that we have some negative values in the BILL_AMT(X) variables, can this be possible?
# Let's count how many we have.
sum = 0
for(i in 12:17){
  sum <- sum + sum(credit[,i] < 0)
}
print(sum)
# Cesc: We have 3932 different negative values in the BILL_AMT(X) set of variables. What should
# we do about them?

# Paco: To deal with negative values, we use log modulus transformation => L(X)=sign(x)*log(|x|+1)
# in the variable, like this

credit.log<-log.modulus(credit,5)
grid.plot(credit.log,15)

initial.histogram(credit,BILL_AMT1,FALSE)
initial.boxplot(credit,BILL_AMT1,FALSE)

###### PAY_AMT(X) ######
# All the values for PAY_AMT(X) are either 0 or positive, which is correct. However, we observe that the distribution
# is very skewed, which leads us to apply logarithms. 
# For example, PAY_AMT1 has a mean of 5664, but its maximum is 873552. We can't know for sure if this is
# correct. Let's take a look at the histogram and the boxplot
ggplot(credit, aes(x = 0, y = PAY_AMT1)) +
  geom_boxplot()

ggplot(credit, aes(x = (PAY_AMT1))) +
  geom_histogram(bins = 20)
ggplot(credit, aes(x = log10(PAY_AMT1))) +
  geom_histogram(bins = 20)

# Paco: nota para Cesc: para análisis de datos multivariantes, Tomás mencionó que el boxplot ya no era tan 
# explicativo, usabamos mahalanobis distance, creo que debemos dejarlo y  fijate en el codigo que uso abajo para 
# detectar outliers con el método lofactor, esto lo preguntamos a Tomás o a Lluis por mail, dime que te parece y 
# lo hacemos.

# Nota para Cesc: cesc ya le he preguntado a Tomás, me dice que en este caso lo mas recomendable es dejar las
# variables sin transformar, ademas a pesar de tener el oulier no nos sirve de nada, es como tu mismo pusiste, solo
# gente con mucha pasta jejeje

##################################################################################################################################

#***************************************************************************#
#                        2.3 check distribution of data                     #
#***************************************************************************#

# Let's check the distribution of all the variables. For the continuous ones we can plot an histogram, 
# for the categorical ones, a barplot with the distribution within the levels of the variable.
grid.plot(credit,15)
# with this plot, we can see that the continuous data is very skewed, and not normal at all, we will apply some 
# transformations to make the data more "normal"

# First, apply log modulus transformation in an attempt to normalize data and then plot
credit.log<-log.modulus(credit,5)
grid.plot(credit.log,15)

# Just Isolating each data and show the difference after and before plotting
grid.plot.continuos(credit.continuos, "histogram")
grid.plot.continuos(credit.log[,-factor.indexes],"histogram")

# Most of the data tends to be normal after the log.modulus transform

# We will update our continuous plot then 
credit.continuos.log <- credit.log[,-factor.indexes]

#***************************************************************************#
#                            2.4 Outlier Detection                          #
#***************************************************************************#

#**************************** Outlier detection with lofactor (Local Outlier Factor) ***********************************
#outlier detection with lofactor (Local Outlier Factor), takes a while 
require(DMwR)
outlier.scores <- lofactor(credit.continuos.log[,-2], k=10)
#we cannot plot, there are NaN, infinite values, possible cause is to have more repeated values than neighbours k
plot(density(outlier.scores))
#pick top 5 as outliers
outliers <- order(outlier.scores, decreasing=T)[1:5]
hist(outliers)
#Which are the outliers?
print(outliers)
# we create a table of scores and id, to remove the supossed outliers
scores <- cbind.data.frame(score = outlier.scores,id = rownames(credit.continuos))
credit.continuos.log.lofactor <- credit.continuos.log[-as.numeric(scores[scores$score >= scores[outliers[5],]$score,]$id)]

#**************************** Outlier detection Mahalanobis ***********************************
#outlier detection with mahalanobis 
require(chemometrics)
outlier.scores2 <- Moutlier(credit.continuos.log[,-2],quantile=0.975,plot=TRUE)
credit.continuos.log.Moutlier <- subset(credit.continuos.log,outlier.scores2$md<outlier.scores2$cutoff)

# Altought, from all this, in some cases, the're could be just rich people in some variables, 
# or really indebted people in others

#***************************************************************************#
#              2.5 Detection of most correlated variables                   #
#***************************************************************************#
cor(credit.continuos)
credit.continuos.log<-credit.log[,-factor.indexes]
library(corrplot)
corrplot(cor(credit.continuos), method="circle")
corrplot(cor(credit.continuos.log), method="circle")
corrplot(cor(credit.continuos.log), method="number")
corrplot(cor(credit.continuos.log), method="color")
corrplot(cor(credit.continuos.log), method="shade")
mosthighlycorrelated(credit.continuos,10)
mosthighlycorrelated(credit.continuos.log,10)

# from the correlation calculus, we can see that there is a clear relationship between the values of BILL_AMT(x),
# and BILL_AMT(x+1), so we can apply a dimensionality reduction technique, like pca for example, on this values

# description of each continuos index with respect to default payment
describeBy(credit.continuos, credit$default.payment.next.month)
pairs(credit.continuos, main = "Default payment pair plot", col = (1:length(levels(credit$default.payment.next.month)))[unclass(credit$default.payment.next.month)])

#***************************************************************************#
#       2.6 Detection of (correlation) of categorical variables             #
#***************************************************************************#

#***************************************************************************#
#                                2.6.1 Sex                                  #
#***************************************************************************#

# In this part we are going to analyze SEX variable from the dataset to see if there is anything interesting
# that gives us more information.

# How many males and females do we have?
initial.barplot(credit,SEX)
# We have 34 % more females than males in our dataset.

# How many Default's and Not-Defaults's do we have for each sex?
grouped.count.plot(credit,SEX,default.payment.next.month)

# It seems that females tend to have less default payments, 
# lets compute the exact proportion to see if there is some kind of bias.
freq.table <- (with(data = credit, table(SEX, default.payment.next.month)))
p.table <- round(prop.table(freq.table, margin = 1), digits = 3)
cbind(freq.table, p.table)
# As we see, the proportion of males with default payment is 0.242, and the proportion of females is 0.208.
# Indeed, males in general have a higher tendency of default payment.


#***************************************************************************#
#                            2.6.2 Education                                #
#***************************************************************************#

# a count of all the values to get an initial idea
initial.barplot(credit,EDUCATION)
# we can see that the 4 "unknown" values are very few, comparing them with the others
# also university is most present in this data, so a better way to see this is to group them all.

# a count check of all the education respect to default payment
grouped.count.plot(credit,EDUCATION,default.payment.next.month)
# university has the most population in both cases, but the tendency is to be not default,
# so a prior assumption will be that university level koreans will be unable to fill their
# debt obligations on time

# Again a check of the proportions will be useful
freq.table <- table(credit$EDUCATION, credit$default.payment.next.month)
p.table <- round(prop.table(freq.table, margin = 1), digits = 3)
cbind(freq.table, p.table)
# The data have interesting results, first showing that not default are the most likely case in
# the unknown categories, and also showing that even if university is the bigest tendency,
# graduate level koreans are the ones that in proportion tend to be unable to fullfill their
# debt obligations in time

#***************************************************************************#
#                            2.6.3 Marriage                                 #
#***************************************************************************#
# How are the levels of the variable distributed?
initial.barplot(credit,MARRIAGE)

# Basically we have 'Married' and 'Single' individuals, here we have the percentages of each type
round(prop.table(table(credit$MARRIAGE)) * 100, digits = 1)

# How many Default's and Not-Defaults's do we have for each type of marriage?
grouped.count.plot(credit,MARRIAGE,default.payment.next.month)

# Let's compute the exact proportion for each level to see if there is some kind of bias.
freq.table <- (with(data = credit, table(MARRIAGE, default.payment.next.month)))
p.table <- round(prop.table(freq.table, margin = 1), digits = 3)
cbind(freq.table, p.table)

# As we can see, the proportion of 'Married' individuals with default payement is 0.235, while the proportion of 
# 'Single' is 0.209. 'Married' individuals have a higher tendency of default payement. 'Other' has a very low percentage 
# of default payment, but we just have 54 individuals, which is not enough data. 'Divorced' has the higher 
# percentage of default, but again we just have 323 individuals, compared to the +20000 rows that 
# are either 'Married' or 'Single'.

#***************************************************************************#
#                                 2.6.3 Age                                 #
#***************************************************************************#
# Even AGE is not categorical, we wanted to do an analysis to check how the age are related to the 
# default or not default category
# a count of all the values to get an initial idea
initial.barplot(credit,AGE)

require(plyr)
head(arrange(as.data.frame(table(credit$AGE)),desc(Freq)), n = 5)
# from this analysis, it is obvious that the quantity of users of credit cards, are centered around
# 29 years old

# Again a check of the proportions will be useful
freq.table <- table(credit$AGE, credit$default.payment.next.month)
p.table <- round(prop.table(freq.table, margin = 1), digits = 3)
(age.df<-as.data.frame(cbind(age=row.names(p.table),freq.table, p.table)))
head(arrange(age.df,desc(age.df$`Not default`)), n = 5)
# in this case, the proportions for not default are around 31 years, somewhat closed from the total
# around 29, but for 57 years, there is also a higher ratio in here, the default parameter is dominant
# in all the cases
head(arrange(age.df,desc(age.df$Default)), n = 5)
# this ratio is somewhat scattered, but they are around 51 and 64 years old, the most default is 
# predominant.

#*****************************************************************************************#
#                             2.5 EDA AGE AND EDUCATION                                   #
#*****************************************************************************************#


#*****************************************************************************************#
#                               Definition of clusters                                    #
#*****************************************************************************************#



#*****************************************************************************************#
#                            3. DERIVATION OF NEW VARIABLES                               #
#*****************************************************************************************#


# Feature extraction/selection
credit.PCA <- PCA(credit,)


#*****************************************************************************************#
#                              Initial model assumptions                                  #
#*****************************************************************************************#
n.rows <- nrow(credit)
n.cols <- ncol(credit)

#initial linear model -> this must be reduced either by factor or dimensionality
credit.lm<-lm(default.payment.next.month~.,data = credit)
plot(credit[,-10],credit[,10])
abline(credit.lm)

# get just one third for validation, the rest to train
test.indexes <- sample(1:n.rows,size = floor(n.rows*0.3),replace = FALSE)
credit.test <- credit[test.indexes,]
credit.train <- credit[-test.indexes,]

# then we check if the test and train samples are balanced
table(credit$default.payment.next.month)

# method to do cross validation for tunning

#array for the best parameters
c.best <- c()
epsilon.best <- c()
gamma.best<-c()
polynomial.degree.best<-c()

#array for computation time
compu.time<- c()

# use svm
model2 <- svm(credit.train[,-25],credit.train[,25],epsilon=0.01,gamma=200, C=100)
lines(credit.train[,-25],predict(model2,credit.train[,-25]),col="green")
credit.svm<-ksvm(credit.train[,-25],credit.train[,25],epsilon=0.01, C=100)


library(rpart)
library(rpart.plot)
library(rattle)
p2 = rpart(default.payment.next.month ~ ., data=credit, control=rpart.control(cp=0.001, xval=10))
p2
plot(p2)

# THE SEQUENCE OF TREES WITH THEIR COMPLEXITY PARAMETER AND COMPUTED ERROR IN THE TRAINING SAMPLE AND BY CROSSVALIDATION
printcp(p2)

plot(p2$cptable[,2],p2$cptable[,3],type="l",xlab="size of the tree",ylab="Relative impurity",main="R(t)")
lines(p2$cptable[,2],p2$cptable[,4],col="blue")
legend("topright",c("R(T)training","R(T)cv"),col=c("black","blue"),lty=1)
plotcp(p2)
