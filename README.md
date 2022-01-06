# pml
---
title: "Practical Machine Learning Assignment Write-up"
author: "Ed Wong"
date: "1/3/2022"
output: html_document
---
View the HTML output of the R Markdown report here
https://htmlpreview.GitHub.io/? https://github.com/code-head/pml/blob/main/pml.html

---

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

## Data 

### Training data 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

### Test data

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

### Source

PUC-Rio http://groupware.les.inf.puc-rio.br/har

## Objective 

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases. 

## Approach

We will first review the data for applicability, and clean-up / trim-down will be performed.  Then the data will be partitioned into a training set and a testing set.  A number of different models will be trained and run against the testing set.  The model with the lowest error rate will be used for the final production of prediction results.


```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Required libraries

First we load the libraries needed for this exercise.

```{r}
library(caret)
library(rpart)
```

Before we begin, we'll set the random seed so the results are reproducible.

```{r}
set.seed(9527)
```

## Loading and cleaning up the data

First we load the raw training set and testing set data files from the URLs

```{r}
raw_train_data <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"))
raw_test_data <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"))
```

Let's do a quick check

```{r}
dim(raw_train_data)
dim(raw_test_data)
```

We could review the raw data in R at this point to get a sense of what kind of data the files contain; however, we did that offline instead in a text editor.


Both the training set and the testing set contain the same variables, except that the last column of the *training* set contains the labels (or the "classe" variable)

```{r}
head(raw_train_data[,c(157:160)])
```

And the last column of the *testing* set contains a dummy problem ID

```{r}
head(raw_test_data[,c(157:160)])
```

The first 7 columns are related to the participants of the study and timestamps for data collection.  

```{r}
head(raw_train_data[,c(1:7)])
head(raw_test_data[,c(1:7)])
```

Those are not needed and will be removed.  The working sets will be called *tmp_train_data* and *tmp_test_data*

```{r}
tmp_train_data <- raw_train_data[,-c(1:7)]
tmp_test_data <- raw_test_data[,-c(1:7)]
dim(tmp_train_data)
dim(tmp_test_data)
```

Next we do a Near-Zero-Variance check for columns that we do not want to include when we train our models

```{r}
nzv_cols <- nearZeroVar(tmp_train_data, saveMetrics = TRUE)
```

Looks like we do have a number of near-zero-variance columns.  Let's remove them from both *tmp_train_data* and *tmp_test_data*.

```{r}
head(nzv_cols)
tmp_train_data <- tmp_train_data[, nzv_cols$nzv==FALSE]
dim(tmp_train_data)
tmp_test_data <- tmp_test_data[, nzv_cols$nzv==FALSE]
dim(tmp_test_data)
```

The files also contained a lot of N/A values.  We will remove columns that contain N/A values as well.

```{r}
na_cols <- apply(tmp_train_data, 2, function(x){any(is.na(x))})
na_cols
train_data <- tmp_train_data[, !na_cols]
dim(train_data)
test_data <- tmp_test_data[, !na_cols]
dim(test_data)
```

We end up with 53 columns which will be much less computationally intensive to train compared with the original 160.


## Partitioning the training set 

To build our models we'll use roughly two thirds of the data for training and the rest for testing.  For simplicity, we'll just name the partitions *trn* and *tst*.

```{r}
train_rows = createDataPartition(y=train_data$classe, p=0.66, list=FALSE)
trn <- train_data[train_rows,]
dim(trn)
tst <- train_data[-train_rows,]
dim(tst)
```


## Creating the prediction models



### Random Forest

We will train and create a model using the Random Forest algorithm (this step takes a while to complete).

```{r}
mod_rf <- randomForest::randomForest(as.factor(classe)~., data=trn)
mod_rf
```

Then we validate the model by performing prediction on the test set.

```{r}
predict_rf <- predict(mod_rf, tst, type="class")
```

Using the Confusion Matrix function, we check the model's accuracy.

```{r}
cm_rf <- confusionMatrix(predict_rf, as.factor(tst$classe))
cm_rf
```

The accuracy is over 99% which is already very acceptable for the type of prediction being performed for this exercise.  Let's plot the model in a graph.

```{r}
plot(mod_rf, main = "Random Forest model")
```

We can see the Random Forest algo used in this prediction model becomes very accurate starting from 100 trees.


### Decision Tree

We will train and create a model using the Decision Tree algorithm (this step takes a while to complete).

```{r}
mod_dt <- rpart(as.factor(classe)~., data=trn, method="class")
rpart.plot::prp(mod_dt)
```

Then we validate the model by performing prediction on the test set.

```{r}
predict_dt <- predict(mod_dt, tst, type="class")
```

Using the Confusion Matrix function, we check the model's accuracy.

```{r}
cm_dt <- confusionMatrix(predict_dt, as.factor(tst$classe))
cm_dt
```

The Decision Tree's accuracy is around 70%.


## Final prediction

For the final prediction, we will be using the Random Forest model built earlier as it yields the smallest error rate.  We'll predict the provided testing set using that model.

```{r}
final_prediction <- predict(mod_rf, test_data, type="class")
final_prediction
```

This concludes Practical Machine Learning assignment write-up.

