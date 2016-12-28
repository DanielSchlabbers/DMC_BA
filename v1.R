# Business Analytics
# Data Mining Cup Introduction
#
# Please note, that this script only has the nature of proposal. It provides useful functions for the steps of data mining but does not cover all possibilities.

# The caret package is used (http://topepo.github.io/caret/index.html)
#install.packages("caret")
library(caret)

#clear environment variables
rm(list=ls())

# For reasons of traceability you must use a fixed seed
set.seed(42) # do NOT CHANGE this seed


######################################################
# 1. Build a Team in the DMC Manager
# https://dmc.dss.in.tum.de/dmc/
# Login with TUM login data ("TUM-Kennung")
#
# Found or join a team (size: 1-4 students)


######################################################
# 2. Load & Explore the Training Data Set
training_data = read.csv("training_yzCDTHz.csv", sep=",")

# Show the structure
str(training_data)

# Size
nrow(training_data)   #4000 rows
ncol(training_data)   #19 columns

# Show the first and last rows
head(training_data)
tail(training_data)

# Show columns with missing values
colSums(is.na(training_data))
# Job 19
# Education 169
# Communication 902
# Outcome 3042

# Explore the class column
table(training_data$CarInsurance)
# 2369 no success 1604 success - balanced dataset


# Explore the balance column
mean(training_data$Balance)
# mean without N/A values
mean(training_data$Balance, na.rm=TRUE)

aggregate(x=training_data$Balance, by=list(training_data$CarInsurance), FUN=mean, na.rm=TRUE)
# more balance -> more likely to buy car insurance

hist(training_data$Balance)
boxplot(training_data$Balance ~ training_data$CarInsurance)


######################################################
# 3. Data Preparation
# (using both training and test data)
# do NOT DELETE any instances in the test data
test_data = read.csv("pub_ZaGS0Z2.csv", sep=",")

# Rename columns - not needed
names(training_data)
#names(training_data)[names(training_data) == "od"] = "order_date"
#names(test_data)[names(test_data) == "od"] = "order_date"


# Nominal attributes
# Job, Education, HHInsurance, CarLoan, Communication, Outcome

#View Distribution
table(training_data$Marital)
table(training_data$Job)
table(training_data$Education)
table(training_data$HHInsurance)
table(training_data$CarLoan)
table(training_data$Communication)
table(training_data$Outcome)
table(training_data$Default)

#Set as nominal attribute
training_data$Marital = factor(training_data$Marital)
training_data$Job = factor(training_data$Job)
training_data$Education = factor(training_data$Education)
training_data$HHInsurance = factor(training_data$HHInsurance)
training_data$CarLoan = factor(training_data$CarLoan)
training_data$Communication = factor(training_data$Communication )
training_data$Outcome = factor(training_data$Outcome)
training_data$Default = factor(training_data$Default)
# CarInsurance is also nominal
training_data$CarInsurance = factor(training_data$CarInsurance)

# set same for test data
test_data$Marital = factor(test_data$Marital)
test_data$Job = factor(test_data$Job)
test_data$Education = factor(test_data$Education)
test_data$HHInsurance = factor(test_data$HHInsurance)
test_data$CarLoan = factor(test_data$CarLoan)
test_data$Communication = factor(test_data$Communication )
test_data$Outcome = factor(test_data$Outcome)
test_data$Default = factor(test_data$Default)


# If a nominal or ordinal column in the test data set contains more levels than the corresponding column in the training data set, you can add levels to the column in the training data set manually using the following command:
#training_data$salutation = factor(training_data$salutation, levels=c(levels(training_data$salutation), "Family"))


# Date attributes -> adds current date can be ignored -> only time difference makes sense
date_format = "%H:%M:%OS"
training_data$CallStart = strptime(training_data$CallStart, date_format)
test_data$CallStart = strptime(test_data$CallStart, date_format)

training_data$CallEnd = strptime(training_data$CallEnd, date_format)
test_data$CallEnd = strptime(test_data$CallEnd, date_format)


# Calculate new column "CallTime" as difference of end and start
training_data$CallTime = as.numeric(training_data$CallEnd - training_data$CallStart)
test_data$CallTime = as.numeric(test_data$CallEnd - test_data$CallStart)


hist(training_data$CallTime)  # a lot of short/non existent calls
table(training_data$CallTime, useNA="ifany")




# Binning/Discretization
#install.packages("arules")
library(arules)



# Multicollinearity


# another package for binning/discretization would be the "discretization" package
# some classifiers use built-in supervised binning, i.e. entropy-based binning



# Feature Selection
#install.packages("FSelector")
library(FSelector)

# delete useless call dates
training_data$Id=NULL
training_data$CallEnd=NULL
training_data$CallStart=NULL
test_data$CallEnd=NULL
test_data$CallStart=NULL

# Calculate weights for the attributes using Info Gain and Gain Ratio
weights_info_gain = information.gain(CarInsurance ~ ., data=training_data)
weights_info_gain
weights_gain_ratio = gain.ratio(CarInsurance ~ ., data=training_data)
weights_gain_ratio

# Select the most important attributes based on Gain Ratio
most_important_attributes <- cutoff.k(weights_gain_ratio, 12)
most_important_attributes
formula_with_most_important_attributes <- as.simple.formula(most_important_attributes, "CarInsurance")
formula_with_most_important_attributes
#Create formula manually
# formula_with_most_important_attributes= return_shipment~delivery_time_discret_ef+state+size+salutation+order_date_weekday

######################################################
# 4. Training & Evaluation
# 3 x 5-fold cross validation
library(RWeka)

fitCtrl = trainControl(method="repeatedcv", number=10, repeats=3)

# information about decision tree parameters
getModelInfo()$J48$parameters

# training a decision tree with specific parameters using the metric "Accuracy"
#install.packages("e1071")
library(e1071)
modelDT = train(formula_with_most_important_attributes, data=training_data, method="J48",
                tuneGrid=data.frame(C=c(0.1, 0.2, 0.3),M=c(2,2,2)),na.action = na.pass)

# training a decision tree, one rule and boosting models using the metric "Accuracy"
#install.packages("caTools")
#install.packages("rocc")
library(caTools)
library(rocc)
modelDT = train(formula_with_most_important_attributes, data=training_data, method="J48", trControl=fitCtrl, metric="Accuracy",na.action = na.pass)
modelOneR = train(formula_with_most_important_attributes, data=training_data, method="OneR", trControl=fitCtrl, metric="Accuracy",na.action = na.pass)
modelBoost = train(formula_with_most_important_attributes, data=training_data, method="LogitBoost", trControl=fitCtrl, metric="Accuracy",na.action = na.pass)
modelRoc = train(formula_with_most_important_attributes, data=training_data, method="rocc", trControl=fitCtrl, metric="Accuracy",na.action = na.pass)
# slow....
#modelRandomForest = train(formula_with_most_important_attributes, data=training_data, method="cforest", trControl=fitCtrl, metric="Accuracy",na.action = na.pass)

# Show results and metrics
modelOneR
modelOneR$results


# Show decision tree
modelDT$finalModel

# Compare results of different models
res = resamples(list(dt=modelDT,oneR=modelOneR, boost =modelBoost, roc =modelRoc))
#res = resamples(list(dt=modelDT,oneR=modelOneR, boost =modelBoost, ranFor =modelRandomForest))
summary(res)

# Show confusion matrix (in percent)
confusionMatrix(modelDT)
confusionMatrix(modelBoost)
confusionMatrix(modelOneR)
confusionMatrix(modelRoc)
#confusionMatrix(modelRandomForest)


######################################################
# 5. Predict Classes in Test Data
prediction_classes = predict.train(object=modelDT, newdata=test_data, na.action=na.pass)
predictions = data.frame(id=test_data$Id, prediction=prediction_classes)
predictions


######################################################
# 6. Export the Predictions
write.csv(predictions, file="predictions_group_name_number.csv", row.names=FALSE)


######################################################
# 7. Upload the Predictions and the Corresponding R Script on DMC Manager
# https://dmc.dss.in.tum.de/dmc/
# Login with TUM login data ("TUM-Kennung")
#
# Maxium number of submissions: 10
#
# Possible errors that could occur:
# - Wrong column names
# - Unknown IDs (if not in Test Data)
# - Missing IDs (if in Test Data but not in Predictions)
# - Wrong file format
