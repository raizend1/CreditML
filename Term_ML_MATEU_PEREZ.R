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
# We can't find any value expressed as 'NA', but we can't know for the moment if there is another type of encoding
# for the missing values. 

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
summary(credit[-factor.indexes])

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
  # Note: some of the values are negative, so log is not an option, needs normalization
  switch(type,
         histogram={for(i in 1:l.data){hist(input.data[,i],main = names(input.data)[i],prob=TRUE);lines(density(input.data[,i]),col="blue", lwd=2)}},
         # histogram={out.plot <- lapply(1:14, function(i) ggplot(data=input.data, aes(input.data[,i])) +
         #                                 geom_histogram(aes(y =..density..),
         #                                                breaks=seq(20, 50, by = 2),
         #                                                col="red",
         #                                                fill="green",
         #                                                alpha = .2) +
         #                                 geom_density(col=i) +
         #                                 labs(title=names(input.data)[i],x=element_blank()))},
         plot={for(i in 1:l.data){plot(input.data[,i],main = names(input.data)[i])}},
         stop("Valid plot types are 'histogram', 'plot'"))
  #marrangeGrob(input.data, nrow=rounded, ncol=rounded)
  #set values to default
  par(mar= c(5, 4, 4, 2))
  par(mfrow=c(1,1))
}

#most of the data is not normal, have some very high skewed values, also the scales are radicall different

credit.continuos<-credit[,-factor.indexes]
credit.factors<-credit[,factor.indexes]

# function to norm
norm.function <- function(x) {(x - min(x, na.rm=TRUE))/((max(x,na.rm=TRUE) - min(x, na.rm=TRUE)))}

# BILL_AMT1<-lapply(credit.continuos[,"BILL_AMT1"],norm.function)
# hist(log(BILL_AMT1),prob=TRUE)
# lines(density(log10(BILL_AMT1)),col="blue")

# check the distribution of all continuos data
summary(credit.continuos)
# BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6, PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6
# are pretty skewed, their mean and median are significant different

draw.plot(credit.continuos,"histogram")

ggplot(data = credit, mapping = aes(x = AGE, ..count..)) + 
  geom_bar(mapping = aes(fill = AGE), position = "dodge") 

ggplot(data = credit, mapping = aes(x = credit[,-factor.indexes],..count..)) + 
  geom_bar()

#Normalize the data

# subset of payment history to check some interesting data - maybe
data.sub.payment.history<-credit[,c(7:12)]

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
# also university is most present in this data

# a count check of all the education respect to default payment
ggplot(credit, aes(x=default.payment.next.month)) +
  geom_bar(mapping = aes(fill = EDUCATION),position="dodge") +
  geom_text(stat='count',aes(label=..count..),vjust=-1)

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
#                                  2.4 EDA PCA                                            #
#*****************************************************************************************#

<<<<<<< HEAD
#*****************************************************************************************#
#                            3. DERIVATION OF NEW VARIABLES                               #
#*****************************************************************************************#

=======
# Feature extraction/selection
credit.PCA <- PCA(credit)
>>>>>>> f081028badd79d09148552b2f4b2dafd0dd2a3b0

#*****************************************************************************************#
#                              Initial model assumptions                                  #
#*****************************************************************************************#
n.rows <- nrow(credit)
n.cols <- ncol(credit)

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



