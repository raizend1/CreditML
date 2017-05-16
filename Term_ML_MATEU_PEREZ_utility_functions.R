#############################################################################################
#                               Utility Functions
#
# Script containing all the function that are common among the files of the project
# The description of each one of them is added in it's description header
#
# Date: 
# Cesc Mateu
# cesc.mateu@gmail.com
# Francisco Perez
# pacogppl@gmail.com
#############################################################################################
#' @title Function to paste without spaces
#'
#' @description
#' Function to paste without spaces
#' @param ...: Strings to paste
#' 
#' @return pasted string with no spaces
#' 
#' @examples
glue<-function(...){paste(...,sep="")}

#' @title Draw a histogram of the variable using ggplot
#'
#' @description
#' This function plot the frequency of the given variable.
#' @param input.dataset: The variable used as input for plot
#' @param input.variable: The variable used as input for plot
#' @param log: Boolean indicating if it log transformation is used
#' 
#' @return Outputs a plot of the variable
#' 
#' @examples
initial.histogram<-function(input.dataset,input.variable,log){
  require(ggplot2)
  if(log)
    eval(parse(text=glue("ggplot(",deparse(substitute(input.dataset)),", aes(x = log(",deparse(substitute(input.variable)),"))) + geom_histogram(bins = 15)")))
  else
    eval(parse(text=glue("ggplot(",deparse(substitute(input.dataset)),", aes(x = ",deparse(substitute(input.variable)),")) + geom_histogram(bins = 15)")))
}


#' @title Draw a boxplot of the variable using ggplot
#'
#' @description
#' This function creates a boxplot of the variable.
#' @param input.dataset: The variable used as input for plot
#' @param input.variable: The variable used as input for plot
#' 
#' @return Outputs a boxplot of the variable
#' 
#' @examples
initial.boxplot<-function(input.dataset,input.variable,log){
  require(ggplot2)
  if(log)
    eval(parse(text=glue("ggplot(",deparse(substitute(input.dataset)),", aes(x = 0, y= log(",deparse(substitute(input.variable)),"))) + geom_boxplot()")))
  else
    eval(parse(text=glue("ggplot(",deparse(substitute(input.dataset)),", aes(x = 0, y = ",deparse(substitute(input.variable)),")) + geom_boxplot()")))
}

#' @title Draw a barplot of the variable using ggplot
#'
#' @description
#' This function creates a boxplot of the variable.
#' @param input.dataset: The variable used as input for plot
#' @param mapping.variable: The variable used mapped for the input
#' 
#' @return Outputs a boxplot of the variable
#' 
#' @examples
initial.barplot<-function(input.dataset,mapping.variable){
  require(ggplot2)
  ggplot(data = input.dataset, aes_string(x = deparse(substitute(mapping.variable)))) + 
    geom_bar(position="dodge", colour="black") +  
    geom_text(stat='count',aes(label=..count..),vjust=-1)
}

#' @title Draw a grid plot of all the variable using ggplot
#'
#' @description
#' This function creates a consolidated plot of all the variables.
#' @param input.dataset: The variable used as input for plot
#' @param bins: Number of bins to use in the histogram
#'  
#' @return Outputs a plot of all the variables
#' 
#' @examples
grid.plot<-function(input.dataset,bins){
  require(ggplot2)
  require(gridExtra)
  par(mfrow=c(1,1))
  #dev.off()
  par(mar=c(3,3,2,2))
  l.data<-length(input.dataset)
  rounded<-round(sqrt(l.data),0)
  out.command<-NULL
  for (i in 1:ncol(input.dataset)){
    if(is.factor(input.dataset[,i])){
      out.command<-glue(out.command,"ggplot(data = ",deparse(substitute(input.dataset)),", mapping = aes(x =",colnames(input.dataset[i]),")) + geom_bar()",",")
    }else{
      out.command<-glue(out.command,"ggplot(data = ",deparse(substitute(input.dataset)),", mapping = aes(x =",colnames(input.dataset[i]),")) + geom_histogram(bins =",bins,")",",")
    }
  }
  out.command<-substring(out.command, 1,nchar(out.command)-1)
  eval(parse(text=glue("final.plot<-arrangeGrob(",out.command,",ncol =", rounded,", nrow =", rounded,")")))
  grid::grid.draw(final.plot)
}

#' @title Draw a gridplot of all the continuos variables
#'
#' @description
#' This function creates a consolidated plot of all the continuos variables.
#' @param input.data: The variable used as input for plot
#' @param type: type of plot, there are histogram and plot
#' 
#' @return Outputs a grid plot of all the variables
#' 
#' @examples
grid.plot.continuos<-function(input.data,type){
  l.data<-length(input.data)
  rounded<-round(sqrt(l.data),0)
  par(mar=c(3,3,2,2))
  par(mfrow=c(rounded-1, rounded+1))
  out.plot<-array(dim = l.data)
  switch(type,
         histogram={for(i in 1:l.data){hist(input.data[,i],main = names(input.data)[i],prob=TRUE);lines(density(input.data[,i]),col="blue", lwd=2)}},
         plot={for(i in 1:l.data){plot(input.data[,i],main = names(input.data)[i])}},
         stop("Valid plot types are 'histogram', 'plot'"))
  #set values to default
  par(mar= c(5, 4, 4, 2))
  par(mfrow=c(1,1))
}


#' @title Executes a log modulus transformation 
#'
#' @description
#' Gives a log modulus transformation (log modulus transformation => L(X)=sign(x)*log(|x|+1)) for all the inputted values.
#' @param input.data: The variable used as input 
#' @param exclude: Columns to exclude from the transformation
#'
#' @return log transformation for all the values
#' 
#' @examples
log.modulus<-function(input.data,exclude){
  output.data<-input.data
  for(i in 1:dim(input.data)[2]){
    if(!(i %in% exclude)){
      if(is.numeric(input.data[,i])){
        signs<-ifelse(input.data[,i]<0,-1,1)
        output.data[,i]<-signs*log(unlist(lapply(input.data[,i],abs))+1)
      }
    }
  }
  return(output.data)
}

#' @title Draw a grid plot on How many Default's and Not-Defaults's do we have for each variable
#'
#' @description
#' This function creates a consolidated plot of all the variables.
#' @param input.dataset: The variable used as input for plot
#' @param bins: Number of bins to use in the histogram
#'  
#' @return Outputs a plot of all the variables
#' 
#' @examples
grouped.count.plot<-function(input.dataset,fill,target){
  require(ggplot2)
  ggplot(data = input.dataset, mapping = aes_string(x = deparse(substitute(target)), "..count..")) + 
    geom_bar(mapping = aes_string(fill = deparse(substitute(fill))), position = "dodge")
}

#' @title Draw a grid plot on How many Default's and Not-Defaults's do we have for each variable
#'
#' @description
#' This function creates a consolidated plot of all the variables.
#' @param input.dataset: The variable used as input for plot
#' @param bins: Number of bins to use in the histogram
#'  
#' @return Outputs a plot of all the variables
#' 
#' @examples
grouped.plot<-function(input.dataset,fill,target){
  require(ggplot2)
  ggplot(data = input.dataset, mapping = aes_string(x = deparse(substitute(target)), "..count..")) + 
    geom_bar(mapping = aes_string(fill = deparse(substitute(fill))), position = "dodge")
}



# function to norm
norm.function <- function(x) {(x - min(x, na.rm=TRUE))/((max(x,na.rm=TRUE) - min(x, na.rm=TRUE)))}


#' @title Executes a log modulus transformation 
#'
#' @description
#' Returns a matrix with the mos correlated data.Credits to Lluis A. Belanche
#' @param mydataframe: The variable used as input 
#' @param numtoreport Quantity of the number to display
#' 
#' @return log transformation for all the values
#' 
#' @examples
mosthighlycorrelated <- function(mydataframe,numtoreport)
{
  # find the correlations
  cormatrix <- cor(mydataframe)
  # set the correlations on the diagonal or lower triangle to zero,
  # so they will not be reported as the highest ones:
  diag(cormatrix) <- 0
  cormatrix[lower.tri(cormatrix)] <- 0
  # flatten the matrix into a dataframe for easy sorting
  fm <- as.data.frame(as.table(cormatrix))
  # assign human-friendly names
  names(fm) <- c("First.Variable", "Second.Variable","Correlation")
  # sort and print the top n correlations
  head(fm[order(abs(fm$Correlation),decreasing=TRUE),],n=numtoreport)
}

####################ORIGINAL FUNCTIONS########################################
# Original plotting·
# ggplot(credit, aes(x = 0, y = LIMIT_BAL)) +
#   geom_boxplot()
# 
# ggplot(credit, aes(x = log10(LIMIT_BAL))) +
#   geom_histogram(bins = 15)

#Original deal with negative logs
# credit.minimum<-lapply(credit.continuos,min)
# credit.positives<- credit.continuos
# for(i in 1:dim(credit.continuos)[2]){
#   if(credit.minimum[i]<0){
#     credit.positives[,i] <- credit.continuos[,i]+(as.numeric(credit.minimum[i])*-1)
#   }
# }

# Let's check the distribution of all the variables. For the continuous ones we can plot an histogram, 
# for the categorical ones, a barplot with the distribution within the levels of the variable.
# 
# for (i in 1:ncol(credit)){
#   if(is.factor(credit[,i])){
#     print("categorical")
#     g <- ggplot(data = credit, mapping = aes(x = credit[,i])) +
#       geom_bar() + 
#       ggtitle(colnames(credit[i]))
#     print(g)
#   }else{
#     print("continuous")
#     c <- ggplot(data = credit, mapping = aes(x = credit[,i])) +
#       geom_histogram()+
#       ggtitle(colnames(credit[i]))
#     print(c)
#   }
# }

# Paco: To deal with negative values, we can sum the minimum value to all the values
# in the variable, like this
# credit.minimum<-lapply(credit.continuos,min)
# credit.positives<- credit.continuos
# for(i in 1:dim(credit.continuos)[2]){
#   if(credit.minimum[i]<0){
#     credit.positives[,i] <- credit.continuos[,i]+(as.numeric(credit.minimum[i])*-1)
#   }
# }
