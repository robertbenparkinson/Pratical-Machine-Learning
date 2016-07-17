# Practical  Machine Learning Assignment
Robert Ben Parkinson  
17 July 2016  

#Overview

Every day more and more data driven sports devices, such as Fit Bit, Jawbone and others, come onto the market. These devices generate a huge amount of data. Our goal in this assignment, and in data science in general, is to take this flood of data and turn it into something useful. Our best hope for doing this is turning to various machine learning techniques used to sort and make useful predictions on our actions, behaviors and much more.  

#Course Data

The data used for this project come from this source: http://groupware.les.inf.puc-rio.br/har. They have been very generous in allowing their data to be used for this kind of assignment.


##Load the Data

The data provides comes in the form of two CSV files. The pml-training.csv files is a rather large and daunting file that consists of 160 columns and 19,622 observation from which we will build out training model.

The second file, pml-testing.csv will be used to test our final model. 


```r
if (!file.exists("pml-training.csv")) {
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "pml-training.csv")
}

if (!file.exists("pml-testing.csv")) {
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "pml-testing.csv")
}

pml_training <-read.csv("pml-training.csv", header = TRUE, sep = ",")
pml_testing <- read.csv("pml-testing.csv", header = TRUE, sep = ",")
```

#Libraries
The following write up uses the "caret", "ggplot2", and randomForest packages. All packages can be found on the https://cran.r-project.org/ website. 



```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.2.5
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(ggplot2)
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.2.5
```

```
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## 
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

##Cleaning The Data

one of the biggest considerations in any major data project is finding a way to 'clean the data' into something useful that won't throw out a sea of errors. 

From both the training and testing sets all of the NA's found in the data were replaced by "0". 
From there I used the nearZeroVar function to remove the columns that very nearly empty. This drastically allows us to simplify our model down from 160 columns of data to 60 columns of data

In the next step I removed the following columns in order to prevent any undue correlations in our data sets.

1. $X or the Id column. 
2. $user_name. This prevents any undue connections between the user and the exercises they preform. For example, Carlos does a lot a squats, therefor any exercise Carlos does must be squats. 
3. All of the Time Stamp columns. For example its always a good idea to warm up before lifting weights. Because of this mis-classifications and connections maybe made on the basis of time rather than on the data itself. 

In the end this leaves us with 52 columns of data. All changes are made to both the training and testing data sets to prevent errors later. 



```r
training <- pml_training[, colSums(is.na(pml_training)) == 0] 
nz <- nearZeroVar(training, saveMetrics = TRUE)
training <- training[, nz$nzv==FALSE]
training <- training[, -c(1:7)]

##cuts $x (id)
##user_name
##timestamp columns part_1, part_2 and part_3
##as well as num_window

testing <- pml_testing[, colSums(is.na(pml_testing)) == 0] 
nz <- nearZeroVar(testing, saveMetrics = TRUE)
testing <- testing[, nz$nzv==FALSE]
testing <- testing[, -c(1:7)]
```

##Data Partition
In this step I have partitioned the training data set into two separated data sets called training data and cross_testingdate (not to be confused with the pml-testing.csv). 



```r
set.seed(999)
inTrain <- createDataPartition(training$classe, p=0.7, list=FALSE)
training_data <- training[inTrain,]
cross_testing_data <- training[-inTrain,]
```

#Random Forest 

After a little trial and error, I decided to use the Random Forest Method to build my training model. 



```r
modFitRF <- randomForest(classe ~ ., data = training_data, ntree = 1000)

modFitRF
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training_data, ntree = 1000) 
##                Type of random forest: classification
##                      Number of trees: 1000
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.52%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3905    1    0    0    0 0.0002560164
## B   16 2637    5    0    0 0.0079006772
## C    0   12 2384    0    0 0.0050083472
## D    0    0   28 2223    1 0.0128774423
## E    0    0    2    6 2517 0.0031683168
```

#Cross Testing Results

Our results are pretty decent. Our model accuracy sits at 99.49%





```r
prf <- predict(modFitRF, cross_testing_data)
confusionMatrix(prf, cross_testing_data$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    6    0    0    0
##          B    0 1129    7    0    0
##          C    0    4 1019   12    0
##          D    0    0    0  951    1
##          E    1    0    0    1 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9946          
##                  95% CI : (0.9923, 0.9963)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9931          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9912   0.9932   0.9865   0.9991
## Specificity            0.9986   0.9985   0.9967   0.9998   0.9996
## Pos Pred Value         0.9964   0.9938   0.9845   0.9989   0.9982
## Neg Pred Value         0.9998   0.9979   0.9986   0.9974   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1918   0.1732   0.1616   0.1837
## Detection Prevalence   0.2853   0.1930   0.1759   0.1618   0.1840
## Balanced Accuracy      0.9990   0.9949   0.9949   0.9932   0.9993
```


##Test Predictions

And finally here are my test predictions using the original testing (pml-testing.csv) data.


```r
testPrediction <- predict(modFitRF, testing, type = "class")
testPrediction
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```



#Fin



