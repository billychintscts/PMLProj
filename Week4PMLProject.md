---
title: "Machine Learning Project - Week 4"
author: "Chin Tham Sang"
date: "August 20, 2017"
output:
  html_document: default
  pdf_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Executive Summary

The project is to predict the manner in which they did the exercise. The variables are data 
from accelerometers on the belt, forearm, arm, and dumbell of 6 participants.

 You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.



### Data Pre Processing

- Load the required library, 


```{r loadLib, echo= TRUE}

library(caret)
library(rpart)
library(randomForest)
library(corrplot)

```


```{r dirc, echo= FALSE}

setwd("~")
```

- Download the data

```{r loaddata, echo= TRUE}
 
trainUrl <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
trainFile <- "./data/pml-training.csv"
testFile  <- "./data/pml-testing.csv"
if (!file.exists("./data")) {
  dir.create("./data")
}
if (!file.exists(trainFile)) {
  download.file(trainUrl, destfile=trainFile)
}
if (!file.exists(testFile)) {
  download.file(testUrl, destfile=testFile)
}

```

- Read the data

Read the data into two data frame


```{r ReadData, echo=TRUE}

trainRaw <- read.csv("./data/pml-training.csv")
testRaw <- read.csv("./data/pml-testing.csv")
dim(trainRaw)
dim(testRaw)
 
```

The training data set contains 19622 observations and 160 variables, while the testing data set contains 20 observations and 160 variables. The "classe" variable in the training set is the outcome to predict.

## Clean The Data

Clean the data and get rid of observations with missing values as well as some meaningless variables.


```{r CleanData, echo=TRUE}

sum(complete.cases(trainRaw))
 
```

- remove columns that contain NA missing values.

```{r missingvalue, echo=TRUE}

trainRaw <- trainRaw[, colSums(is.na(trainRaw)) == 0] 
testRaw <- testRaw[, colSums(is.na(testRaw)) == 0] 
 
```

- remove variables not contribted to the accelerometer measurements.

```{r notapplicable, echo=TRUE}

classe <- trainRaw$classe
trainRemove <- grepl("^X|timestamp|window", names(trainRaw))
trainRaw <- trainRaw[, !trainRemove]
trainCleaned <- trainRaw[, sapply(trainRaw, is.numeric)]
trainCleaned$classe <- classe
testRemove <- grepl("^X|timestamp|window", names(testRaw))
testRaw <- testRaw[, !testRemove]
testCleaned <- testRaw[, sapply(testRaw, is.numeric)]
 
```

Now, the cleaned training data set contains 19622 observations and 53 variables, while the testing data set contains 20 observations and 53 variables. The "classe" variable is still in the cleaned training set.

## Slice the data

The data is partition into 60% as training dataset, 40% as testing dataset.
 

```{r datapartition, echo=TRUE}

set.seed(22123) # For reproducibile purpose
inTrain <- createDataPartition(trainCleaned$classe, p=0.60, list=F)
trainData <- trainCleaned[inTrain, ]
testData <- trainCleaned[-inTrain, ]
 
```

## Data Modeling

We fit a predictive model for activity recognition using Random Forest algorithm because it automatically selects important variables and is robust to correlated covariates & outliers in general. We will use 5-fold cross validation when applying the algorithm.

```{r rfmodel, echo=TRUE}

controlRf <- trainControl(method="cv", 5)
modelRf <- train(classe ~ ., data=trainData, method="rf", trControl=controlRf, ntree=100)
modelRf
 
```

## Validation of the model

Then, we estimate the performance of the model on the validation data set.

```{r rfpredict, echo=TRUE}

predictRf <- predict(modelRf, testData)
confusionMatrix(testData$classe, predictRf)
 
```

### Testing the accuracy

```{r raccuracy, echo=TRUE}

accuracy <- postResample(predictRf, testData$classe)
accuracy
 
```

### Estimate out of sample error

```{r ooserr, echo=TRUE}

ooserror <- 1 - as.numeric(confusionMatrix(testData$classe, predictRf)$overall[1])
ooserror
 
```

So, the estimated accuracy of the model is 99.12% and the estimated out-of-sample error is 0.88%.

## Prediction on testdata set

- remove problem_id from testdata set

```{r restest, echo=TRUE}

result <- predict(modelRf, testCleaned[, -length(names(testCleaned))])
result
 
```

## Prediction outcome (answer)

The answer fo each question is writen to problem_id_n.txt, where n is 1 to 20. The value is entered into the Week 4 assignment.

```{r answer, echo=TRUE}
answers <- result
pml_write_files <- function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("./result/problem_id_",i,".txt")
    write.table(x[i], file=filename, quote=FALSE,
                row.names=FALSE, col.names=FALSE)
  }
}
pml_write_files(answers)

```


## Appendix : Figures

### Correlation Matrix Visualization


```{r corrvisual, echo=FALSE}

corrPlot <- cor(trainData[, -length(names(trainData))])
corrplot(corrPlot, method="color" , tl.cex = 0.7) 

```

### Tree Visualization

```{r treevisual, echo=FALSE}
#install.packages("rpart.plot")
library("rpart")
library(rpart.plot)
treeModel <- rpart(classe ~ ., data=trainData, method="class")
#rpart.plot(treeModel)
prp(treeModel)
 
```





