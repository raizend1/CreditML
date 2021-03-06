# About the project
This project uses multivariant methods to discover hidden features, and machine learning techniques to  on the default of credit card clients Data Set, from UCI Machine learning repository. For this process we will use methods from ML from MIRI Masters at UPC.

# Methodology 
The selected language to do the processing is R, with its IDE R Studio. For the plots we will use ggplot and
lattice packages. The methodology is sequential, as seen in Machine Learning classes, we first do an exploratory
analysis of the data to identify how the data is imported into R, the distribution of the values, check for missing
values, apply dimensionality reduction techniques, then split the dataset into train and test subsets to fit the
four diferent models (Logistic Regression, Support Vector Machine, Neural Networks and Random Forest) and
compare their prediction values. For this purposes, we are using the train function from caret[5], to tune the
parameters, using a 10 fold cross validation method, and then passing the optimal parameters to the desired
function.

# Files Content
The preprocessing is done in *Term_ML_MATEU_PEREZ*, *Prediction* contains all the applyed methods and its results, ROC displays the roc curve plot and obtains the AUC of them and *Term_ML_MATEU_PEREZ_utility_functions* contains extra functions used to clean, plot, create latex tables, sample stratify used in this project.
It is also required a *workingDir.r* archive, that follows the next structure:

path <- *root path of the project*
dataDir<-glue(path,"/data")
plotDir<-glue(path,"/plot")
codeDir<-glue(path,"/code")

*glue* is a function included in Term_ML_MATEU_PEREZ_utility_functions that paste without spaces.


# How to use it
In

# More detailed explanation on dataset used
This research employed a binary variable, default
payment (Yes = 1, No = 0), as the response variable. This study reviewed the
literature and used the following 23 variables as explanatory variables:


X1: Amount of the given credit (NT dollar):
it includes both the individual consumer credit and his/her family (supplementary)
credit.


X2: Gender (1 = male; 2 = female).


X3: Education (1 = graduate school; 2 =
university; 3 = high school; 0, 4, 5, 6 = others).


X4: Marital status (1 = married; 2 = single;
3 = divorce; 0=others).


X5: Age (year).


X6 - X11: History of past payment. We tracked
the past monthly payment records (from April to September, 2005) as follows: X6
= the repayment status in September, 2005; X7 = the repayment status in August,
2005; . . .;X11 = the repayment status in April, 2005. The measurement scale
for the repayment status is: 


-2: No consumption; -1: Paid in full; 0: The
use of revolving credit; 1 = payment delay for one month; 2 = payment delay for
two months; . . .; 8 = payment delay for eight months; 9 = payment delay for
nine months and above.

Note:Is it interesting to get which one of each category has?

Note:Un cr?dito revolvente o cr?dito revolving (del ingl?s revolving credit) es un 
tipo de cr?dito que no tiene un n?mero fijo de cuotas, en contraste con el cr?dito convencional. Ejemplos de cr?ditos revolventes son los asociados a las tarjetas de cr?dito.


X12-X17: Amount of bill statement (NT
dollar). X12 = amount of bill statement in September, 2005; X13 = amount of
bill statement in August, 2005; . . .; X17 = amount of bill statement in April,
2005. 

X18-X23: Amount of previous payment (NT
dollar). X18 = amount paid in September, 2005; X19 = amount paid in August,
2005; . . .;X23 = amount paid in April, 2005.


Y:
client's behavior; Y=0 then not default, Y=1 then default

