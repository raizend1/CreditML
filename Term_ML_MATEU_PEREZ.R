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
library(knitcitations)
library(doParallel)
library(gridExtra)
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

# library("caret")
# cl <- makeCluster(detectCores())
# registerDoParallel(cl)
# zero.variance <- nearZeroVar(credit, saveMetrics = TRUE)
# stopCluster(cl)
# str(zero.variance)
# zero.variance[zero.variance[,"zeroVar"] > 0, ] 
# zero.variance[zero.variance[,"zeroVar"] + zero.variance[,"nzv"] > 0, ] 

#There are none, we can conclude that all the predictors are relevant for the moment.

#***************************************************************************#
#                              2.2 Check N/A Values                         #
#***************************************************************************#

# First check N/A values

# which(is.na(credit),arr.ind=TRUE) 
# md.pattern(credit)

# It can't find any value expressed as 'NA', but there are some rows where all the values for the billing
# statements and the previous payment are 0, this could be treated as a missing value, because if there is 
# a credit card issued, there must be values for this columns, so we treat them as missing values. First 
# a check of how many of this occurrences exist is needed
check.new.clients<-function(input.data){
  indexes<-NULL
  j<-1
  for(i in 1:dim(input.data)[1]){
    if((!all(input.data[i,c(1:6)]=="NC")) && all(input.data[i,c(7:dim(input.data)[2])]==0)){
      indexes[j]<- i
      j<-j+1
    }
  }
  return(indexes)
}

check.zero.rows<-function(input.data){
  indexes<-NULL
  j<-1
  for(i in 1:dim(input.data)[1]){
    if((all(input.data[i,c(1:6)]=="NC")) && all(input.data[i,c(7:dim(input.data)[2])]==0)){
      indexes[j]<- i
      j<-j+1
    }
  }
  return(indexes)
}

cl <- makeCluster(detectCores())
registerDoParallel(cl)
num.new.clients<-check.new.clients(credit[,c(6:23)])
num.zeros.index<-check.zero.rows(credit[,c(6:23)])
stopCluster(cl)

#number of "inactive" users
length(num.zeros.index)
round((length(num.zeros.index)*100)/dim(credit)[1],2)

#number of "new" users
length(num.new.clients)
round((length(num.new.clients)*100)/dim(credit)[1],2)

credit<-credit[-num.zeros.index,]

# update continuous and factor data
credit.continuos<-credit[,-factor.indexes]
credit.factors<-credit[,factor.indexes]

################# Analysis of the continuous variables ###################

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

# credit.log<-log.modulus(credit,5)
# grid.plot(credit.log,15)

initial.histogram(credit,BILL_AMT1,FALSE)
initial.boxplot(credit,BILL_AMT1,FALSE)

###### PAY_AMT(X) ######
# All the values for PAY_AMT(X) are either 0 or positive, which is correct. However, we observe that the distribution
# is very skewed, which leads us to apply logarithms. 
# For example, PAY_AMT1 has a mean of 5664, but its maximum is 873552. We can't know for sure if this is
# correct. Let's take a look at the histogram and the boxplot

# ggplot(credit, aes(x = 0, y = PAY_AMT1)) +
#   geom_boxplot()
# ggplot(credit, aes(x = (PAY_AMT1))) +
#   geom_histogram(bins = 20)
# ggplot(credit, aes(x = log10(PAY_AMT1))) +
#   geom_histogram(bins = 20)

# Paco: nota para Cesc: para análisis de datos multivariantes, Tomás mencionó que el boxplot ya no era tan 
# explicativo, usabamos mahalanobis distance, creo que debemos dejarlo y  fijate en el codigo que uso abajo para 
# detectar outliers con el método lofactor, esto lo preguntamos a Tomás o a Lluis por mail, dime que te parece y 
# lo hacemos.

# Nota para Cesc: cesc ya le he preguntado a Tomás, me dice que en este caso lo mas recomendable es dejar las
# variables sin transformar, ademas a pesar de tener el oulier no nos sirve de nada, es como tu mismo pusiste, solo
# gente con mucha pasta jejeje

##################################################################################################################################

#***************************************************************************#
#                        2.3 Check distribution of data                     #
#***************************************************************************#

# Let's check the distribution of all the variables. For the continuous ones we can plot an histogram, 
# for the categorical ones, a barplot with the distribution within the levels of the variable.

grid.plot(credit,15)
save.plot(grid.plot(credit,15),"Variable_Distribution.jpeg","jpeg",plotDir,"1500","1500","110")
setwd(codeDir)

# With this plot, we can see that the continuous data is very skewed, and not normal at all, we will apply some 
# transformations to make the data more "normal"

# We don't need to apply any standardization, all the continuous predictors are in NT Dollars
# But we need to apply log modulus transformation in an attempt to normalize data in terms of their orders of magnitude,
# Then we will plot and check how the data hias changed

credit<-log.modulus(credit,5)
grid.plot(credit,15)
save.plot(grid.plot(credit,15),"Variable_Distribution_Log.jpeg","jpeg",plotDir,"1500","1500","110")

# Just Isolating each data and show the difference after and before plotting

# grid.plot.continuos(credit.continuos, "histogram")
# grid.plot.continuos(credit[,-factor.indexes],"histogram")

# Most of the data tends to be normal after the log.modulus transform

# We will update our continuous plot then 
credit.continuos <- credit[,-factor.indexes]

#***************************************************************************#
#                            2.4 Outlier Detection                          #
#***************************************************************************#

#**************************** Outlier detection with lofactor (Local Outlier Factor) ***********************************
# Outlier detection with lofactor (Local Outlier Factor)

require(DMwR)
outlier.scores <- lofactor(credit.continuos[,-2], k=10) # Warning: This takes a while!

# We cannot plot, there are NaN, infinite values, possible cause is to have more repeated values than neighbours k
plot(density(outlier.scores))

# Pick top 5 as outliers
(outliers <- order(outlier.scores, decreasing=T)[1:5])
hist(outliers)

# Which are the outliers?
print(outliers)

# We create a table of scores and id, to check the "oulier" values
scores <- cbind.data.frame(score = outlier.scores,id = rownames(credit.continuos))

# Credit <- credit[-as.numeric(scores[scores$score >= scores[outliers[5],]$score,]$id)]

# We can take out the outliers, but according to our preprocessing, the values had been already treated with a 
# log modulus transformation, so the outliers are treated and we don't consider appropiate to lose more information
# of the dataset, instead, use a robust method to process the data. 
# From all this, in some cases, they're could be just rich people in some variables, 
# or really indebted people in others

# credit <- credit[-(which(!is.na(scores[scores$score >= scores[outliers[5],]$score,]$id))),]
# credit.continuos <- credit[,-factor.indexes]
# credit.factors <- credit[,factor.indexes]


#***************************************************************************#
#              2.5 Detection of most correlated variables                   #
#***************************************************************************#
require(corrplot)
par(mfrow=c(1,2))
corrplot(cor(credit[,-factor.indexes]), method="circle")
corrplot(cor(credit[,-factor.indexes]), method="number")
par(mfrow=c(1,1))

# Before log modulus: from the correlation calculus, we can see that there is a clear relationship 
# between the values of BILL_AMT(x), and BILL_AMT(x+1), so we can apply a dimensionality reduction technique, 
# like pca for example, on this values.

# After log modulus: there is no evidence of strong correlation between the predictors, BILL_AMT are somewhat
# correlated between them and also the BILL_AMT and PAY_AMT from the same month, we will use PCA to discover further
# lineaar combinations

# description of each continuos index with respect to default payment
describeBy(credit.continuos, credit$default.payment.next.month)
#pairs(credit.continuos, main = "Default payment pair plot", col = (1:length(levels(credit$default.payment.next.month)))[unclass(credit$default.payment.next.month)])

#***************************************************************************#
#       2.6 Detection of (correlation) of categorical variables             #
#***************************************************************************#

#***************************************************************************#
#                                2.6.1 Sex                                  #
#***************************************************************************#

# In this part we are going to analyze SEX variable from the dataset to see if there is anything interesting
# that gives us more information.

par(mfrow=c(1,2))
# How many males and females do we have?

initial.barplot(credit,SEX)

# We have 34 % more females than males in our dataset.

# How many Default's and Not-Defaults's do we have for each sex?

grouped.count.plot(credit,SEX,default.payment.next.month)

# It seems that females tend to have less default payments, 
# lets compute the exact proportion to see if there is some kind of bias.

freq.table <- (with(data = credit, table(SEX, default.payment.next.month)))
p.table <- round(prop.table(freq.table, margin = 1), digits = 3)
(cbind(freq.table, p.table))

create.latex.table(df=cbind(freq.table, p.table),type="latex",caption="Y and Sex frequencies",file=glue(dataDir,"/y_sex.tex"),digits = 2)

# As we see, the proportion of males with default payment is 0.242, and the proportion of females is 0.208.
# Indeed, males in general have a higher tendency of default payment.

#***************************************************************************#
#                            2.6.2 Education                                #
#***************************************************************************#

par(mfrow=c(1,2))
# A count of all the values to get an initial idea

initial.barplot(credit,EDUCATION)

# We can see that the 4 "unknown" values are very few, comparing them with the others
# also university is most present in this data, so a better way to see this is to group them all.

# a count check of all the education respect to default payment

grouped.count.plot(credit,EDUCATION,default.payment.next.month)

# University has the most population in both cases, but the tendency is to be not default,
# so a prior assumption will be that university level koreans will be unable to fill their
# debt obligations on time

# Again a check of the proportions will be useful
freq.table <- table(credit$EDUCATION, credit$default.payment.next.month)
p.table <- round(prop.table(freq.table, margin = 1), digits = 3)
cbind(freq.table, p.table)

create.latex.table(df=cbind(freq.table, p.table),type="latex",caption="Y and Education frequencies",file=glue(dataDir,"/y_education.tex"),digits = 2)

# The data have interesting results, first showing that not default are the most likely case in
# the unknown categories, and also showing that even if university is the bigest tendency,
# graduate level koreans are the ones that in proportion tend to be unable to fullfill their
# debt obligations in time

#***************************************************************************#
#                            2.6.3 Marriage                                 #
#***************************************************************************#

par(mfrow=c(1,2))
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

create.latex.table(df=cbind(freq.table, p.table),type="latex",caption="Y and Marriage frequencies",file=glue(dataDir,"/y_marriage.tex"),digits = 2)

# As we can see, the proportion of 'Married' individuals with default payement is 0.235, while the proportion of 
# 'Single' is 0.209. 'Married' individuals have a higher tendency of default payement. 'Other' has a very low percentage 
# of default payment, but we just have 54 individuals, which is not enough data. 'Divorced' has the higher 
# percentage of default, but again we just have 323 individuals, compared to the +20000 rows that 
# are either 'Married' or 'Single'.

#***************************************************************************#
#                                 2.6.3 Age                                 #
#***************************************************************************#
par(mfrow=c(1,2))

# Even AGE is not categorical, we wanted to do an analysis to check how the age are related to the 
# default or not default category
# A count of all the values to get an initial idea

initial.barplot(credit,AGE)

grouped.count.plot(credit,AGE,default.payment.next.month)

invisible(require(plyr))
head(arrange(as.data.frame(table(credit$AGE)),desc(Freq)), n = 5)

# From this analysis, it is obvious that the quantity of users of credit cards, are centered around
# 29 years old
# Again a check of the proportions will be useful

freq.table <- table(credit$AGE, credit$default.payment.next.month)
p.table <- round(prop.table(freq.table, margin = 1), digits = 3)
(age.df<-as.data.frame(cbind(age=row.names(p.table),freq.table, p.table)))

head(arrange(age.df,desc(age.df$`Not default`)), n = 5)

# In this case, the proportions for not default are around 31 years, somewhat closed from the total
# around 29, but for 57 years, there is also a higher ratio in here, the default parameter is dominant
# in all the cases

head(arrange(age.df,desc(age.df$Default)), n = 5)

# This ratio is somewhat scattered, but they are around 51 and 64 years old, the most default is 
# predominant.

create.latex.table(df=head(arrange(age.df,desc(age.df$`Not default`)), n = 5),type="latex",caption="Y and Age Top 5 Not Default frequencies",file=glue(dataDir,"/y_age_not_default.tex"),digits = 2)

create.latex.table(df=head(arrange(age.df,desc(age.df$Default)), n = 5),type="latex",caption="Y and Age Top 5 Default frequencies",file=glue(dataDir,"/y_age_default.tex"),digits = 2)

#*****************************************************************************************#
#                                    Cluster Analysis                                     #
#*****************************************************************************************#
# simplied method that requiers more computational power, we need to do a stratified sampling before use it
# require(NbClust)
# nb <- NbClust(credit[,-factor.indexes], distance = "euclidean", min.nc = 2,max.nc = 10, method = "complete", index ="all")

#*****************************************************************************************#
#                                 3. PCA Construction                                     #
#*****************************************************************************************#
n<-nrow(credit[,-factor.indexes])
p<-ncol(credit[,-factor.indexes])
require(FactoMineR)
e_ncp<-estim_ncp(credit[,-factor.indexes], ncp.min=0, ncp.max=p-1, scale=TRUE, method="GCV")
(ncp<-e_ncp$ncp)

credit.PCA <- PCA(credit,quali.sup = factor.indexes,ncp = ncp)
summary(credit.PCA)

require(factoextra)
#PCA Colored according to the quality of representation of the variables
fviz_pca_var(credit.PCA, col.var="cos2") +
  scale_color_gradient2(low="white", mid="blue", high="red", midpoint=0.5) + theme_minimal()

# Scores (i.e. principal coordinates) are in res.pca$ind$coord
# The variance of the individuals' coordinates for a dimension corresponds to the eigenvalue of this dimension.

# Loadings (i.e. standard coordinates) are not given by FactoMineR's methods. They return principal coordinates.
# We have to calculate them by dividing variables' coordinates on a dimension by this dimension's eigenvalue's 
# square root.
credit.loadings<-sweep(credit.PCA$var$coord,2,sqrt(credit.PCA$eig[1:ncol(credit.PCA$var$coord),1]),FUN="/")

bill.indexes<-grepl("BILL_AMT", names(credit.continuos))
pay.indexes<-grepl("PAY_AMT", names(credit.continuos))

result<-linear.combination(credit.continuos[,bill.indexes],credit.loadings[bill.indexes,1])

