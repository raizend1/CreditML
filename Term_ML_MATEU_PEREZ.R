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
glue<-function(...){paste(...,sep="")}
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
library(ggrepel)
library(ggthemes)

#***************************************************************************#
#               1. Data Loading and some Preprocessing                      #
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

#***************************************************************************#
#                2. Initial Exploratory Data Analysis (EDA)                 #
#***************************************************************************#

# Remove unnecesary data: ID
credit<- credit[,-1]
factor.indexes<-factor.indexes-1 # update indexes of the factors

# define continuous and factor data
credit.continuos<-credit[,-factor.indexes]
credit.factors<-credit[,factor.indexes]

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

num.zeros.index<-check.zero.rows(credit.continuos[,-c(1:3)])
length(num.zeros.index)
(length(num.zeros.index)*100)/dim(credit)[1]
# so in total there are 795 of this kind of data, it represents 2.65 of the data, but we will discard them anyway
# because this data can produce errors in further steps like the change in scales

credit<-credit[-num.zeros.index,]

# Let's check the distribution of all the variables. For the continuous ones we can plot an histogram, 
# for the categorical ones, a barplot with the distribution within the levels of the variable.

for (i in 1:ncol(credit)){
  if(is.factor(credit[,i])){
    print("categorical")
    g <- ggplot(data = credit, mapping = aes(x = credit[,i])) +
      geom_bar() + 
      ggtitle(colnames(credit[i]))
    print(g)
  }else{
    print("continuous")
    c <- ggplot(data = credit, mapping = aes(x = credit[,i])) +
      geom_histogram()+
      ggtitle(colnames(credit[i]))
    print(c)
  }
}


################# Analysis of the continuous variables ###################
summary(credit.continuos)

###### LIMIT_BAL ######
# Everything seems correct, we have an extreme outlier of an individual with a credit limit of 1.000.000, but it is
# not impossible, just a lucky rich person :).
credit[credit$LIMIT_BAL > 900000,]
ggplot(credit, aes(x = 0, y = LIMIT_BAL)) +
  geom_boxplot()

ggplot(credit, aes(x = log10(LIMIT_BAL))) +
  geom_histogram(bins = 15)

###### AGE ######
# Nothing weird. The mean age of the individuals is 35.49 years.
ggplot(credit, aes(x = 0, y = AGE)) +
  geom_boxplot()

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

# Paco: To deal with negative values, we can sum the minimum value to all the values
# in the variable, like this
credit.minimum<-lapply(credit.continuos,min)
credit.positives<- credit.continuos
for(i in 1:dim(credit.continuos)[2]){
  if(credit.minimum[i]<0){
    credit.positives[,i] <- credit.continuos[,i]+(as.numeric(credit.minimum[i])*-1)
  }
}

# lapply(credit.positives,min)
 
ggplot(credit.positives, aes(x = 0, y = BILL_AMT1)) +
  geom_boxplot()

ggplot(credit.positives, aes(x = (BILL_AMT1))) +
  geom_histogram(bins = 15)

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
  # Note: some of the values are negative, so in that case the credit with the positive values is used
  switch(type,
         histogram={for(i in 1:l.data){hist(input.data[,i],main = names(input.data)[i],prob=TRUE);lines(density(input.data[,i]),col="blue", lwd=2)}},
         # histogram={out.plot <- lapply(1:14, function(i) ggplot(data=input.data, aes(input.data[,i])) +
         #                                 geom_histogram(aes(y =..density..),breaks=seq(20, 50, by = 2),col="red",fill="green",alpha = .2) +
         #                                 geom_density(col=i) +labs(title=names(input.data)[i],x=element_blank()))},
         plot={for(i in 1:l.data){plot(input.data[,i],main = names(input.data)[i])}},
         stop("Valid plot types are 'histogram', 'plot'"))
  #marrangeGrob(input.data, nrow=rounded, ncol=rounded)
  #set values to default
  par(mar= c(5, 4, 4, 2))
  par(mfrow=c(1,1))
}

#draw the joint plot with all "original" values for continuous data
draw.plot(credit.continuos, "histogram")
bx <- boxcox(BILL_AMT1 ~, data = credit.positives,
             lambda = seq(-0.25, 0.25, length = 10))
lambda <- bx$x[which.max(bx$y)]
credit.positives.bc <- (credit.positives$BILL_AMT1^lambda - 1)/lambda
hist(Capital.BC, main="Look at that now!")

#draw the joint plot with all positive values for continuous data
draw.plot(credit.positives, "histogram")

#most of the data is not normal, have some very high skewed values, also the scales are radicall different

# function to norm
norm.function <- function(x) {(x - min(x, na.rm=TRUE))/((max(x,na.rm=TRUE) - min(x, na.rm=TRUE)))}

#**************************** Outlier detection with lofactor (Local Outlier Factor) ***********************************
#outlier detection with lofactor (Local Outlier Factor), takes a while 
outlier.scores <- lofactor(credit.continuos[,-2], k=10)
#we cannot plot, there are NaN, infinite values, possible cause is to have more repeated values than neighbours k
plot(density(outlier.scores))
#pick top 5 as outliers
outliers <- order(outlier.scores, decreasing=T)[1:5]
hist(outliers)
#Which are the outliers?
print(outliers)
# we create a table of scores and id, to remove the supossed outliers
scores <- cbind.data.frame(score = outlier.scores,id = rownames(credit.continuos))
credit.continuos1 <- credit.continuos[-as.numeric(scores[scores$score >= scores[outliers[5],]$score,]$id)]

#**************************** Outlier detection Mahalanobis ***********************************
#outlier detection with mahalanobis 
outlier.scores2 <- outlier(credit.continuos[,-2])
#pick top 5 as outliers
outliers2 <- order(outlier.scores2, decreasing=T)[1:5]
hist(outliers2)
#Which are the outliers?
print(outliers2)
# we create a table of scores and id, to remove the supossed outliers
scores2 <- cbind.data.frame(score = outlier.scores2,id = rownames(credit.continuos))
credit.continuos2 <- credit.continuos[-as.numeric(scores2[scores2$score >= scores2[outliers2[5],]$score,]$id)]

# Altought, from all this, in some cases, the're could be just rich people, or really indebted people

# description of each continuos index with respect to default payment
describeBy(credit.continuos, credit$default.payment.next.month)
pairs(credit.continuos, main = "Default payment pair plot", col = (1:length(levels(credit$default.payment.next.month)))[unclass(credit$default.payment.next.month)])

#***************************************************************************#
#                             2.1 EDA Sex                                   #
#***************************************************************************#

# In this part we are going to analyze SEX variable from the dataset to see if there is anything interesting
# that gives us more information.

# How many males and females do we have?
ggplot(data = credit, mapping = aes(x = SEX)) + 
  geom_bar() +
  geom_text(stat='count',aes(label=..count..),vjust=-1)

# We have 34 % more females than males in our dataset.

# How many Default's and Not-Defaults's do we have for each sex?
ggplot(data = credit, mapping = aes(x = default.payment.next.month, ..count..)) + 
  geom_bar(mapping = aes(fill = SEX), position = "dodge")

# It seems that females tend to have less default payments, 
# lets compute the exact proportion to see if there is some kind of bias.
freq.table <- (with(data = credit, table(SEX, default.payment.next.month)))
p.table <- round(prop.table(freq.table, margin = 1), digits = 3)
cbind(freq.table, p.table)

# As we see, the proportion of males with default payment is 0.242, and the proportion of females is 0.208.
# Indeed, males in general have a higher tendency of default payment.


#*****************************************************************************************#
#                                  2.2 EDA Education                                      #
#*****************************************************************************************#

# a count of all the values to get an initial idea
ggplot(credit, aes(x=EDUCATION)) +
  geom_bar(position="dodge", colour="black") + 
  geom_text(stat='count',aes(label=..count..),vjust=-1)
# we can see that the 4 "unknown" values are very few, comparing them with the others
# also university is most present in this data, so a better way to see this is to group them all.

# a count check of all the education respect to default payment
ggplot(credit, aes(x=default.payment.next.month)) +
  geom_bar(mapping = aes(fill = EDUCATION),position="dodge") +
  geom_text(stat='count',aes(label=..count..),vjust=-1)
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

#*****************************************************************************************#
#                                  2.3 EDA Marriage                                       #
#*****************************************************************************************#

# In this part we are going to analyze the 'MARRIAGE' variable from the dataset to see if there
# is anything interesting that gives us more information.

# How are the levels of the variable distributed?
ggplot(data = credit, mapping = aes(x = MARRIAGE)) + 
  geom_bar() +
  geom_text(stat='count',aes(label=..count..),vjust=-1)

# Basically we have 'Married' and 'Single' individuals, here we have the percentages of each type
round(prop.table(table(credit$MARRIAGE)) * 100, digits = 1)

# How many Default's and Not-Defaults's do we have for each type of marriage?
ggplot(data = credit, mapping = aes(x = default.payment.next.month, ..count..)) + 
  geom_bar(mapping = aes(fill = MARRIAGE), position = "dodge")

# Let's compute the exact proportion for each level to see if there is some kind of bias.
freq.table <- (with(data = credit, table(MARRIAGE, default.payment.next.month)))
p.table <- round(prop.table(freq.table, margin = 1), digits = 3)
cbind(freq.table, p.table)

# As we can see, the proportion of 'Married' individuals with default payement is 0.235, while the proportion of 
# 'Single' is 0.209. 'Married' individuals have a higher tendency of default payement. 'Other' has a very low percentage 
# of default payment, but we just have 54 individuals, which is not enough data. 'Divorced' has the higher 
# percentage of default, but again we just have 323 individuals, compared to the +20000 rows that 
# are either 'Married' or 'Single'.


#*****************************************************************************************#
#                                  2.4 EDA AGE                                            #
#*****************************************************************************************#

# a count of all the values to get an initial idea
ggplot(credit, aes(x=AGE)) +
  geom_bar(position="dodge", colour="black") + 
  geom_text(stat='count',aes(label=..count..),vjust=-1)
require(plyr)
head(arrange(as.data.frame(table(credit$AGE)),desc(Freq)), n = 5)
# from this analysis, it is obvious that the quantity of users of credit cards, are centered around
# 29 years old

# how many by age
ggplot(data = credit, mapping = aes(x = default.payment.next.month, ..count..)) + 
  geom_bar(mapping = aes(fill = AGE), position = "dodge")

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
credit.PCA <- PCA(credit)


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



