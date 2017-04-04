#remove old objects for safety resons
rm(list=ls(all=TRUE))
dev.off()

set.seed(123)

#utility function
glue<-function(...){paste(...,sep="")}

#set initial wd
workingDir <- "D:/Documents/MIRI/Semestre 3/Machine Learning/L06/code"
#workingDir <-"E:/Documents/Mis Documentos/MIRI/Semestre 3/Machine Learning/L06/code"
dataDir<-glue(workingDir,"/data")
plotDir<-glue(workingDir,"/plots")
codeDir<-glue(workingDir,"/code")
setwd(workingDir)

##this is the test for the first commit
