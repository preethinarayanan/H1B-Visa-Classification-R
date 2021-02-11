library("caret")
library("randomForest")
library("dplyr")
library("glmnet")
library("ROCR")
library("DMwR")
library("ROSE")
library("pROC")
library("rpart.plot")

h1b <- read.csv('1. Master H1B Dataset.csv')
h1b <- na.omit(h1b)
str(h1b)

h1b <- subset(h1b, select = -c(CASE_SUBMITTED_DAY, DECISION_DAY, EMPLOYER_NAME, NAICS_CODE, PW_SOURCE, 
                               PW_SOURCE_YEAR, PW_SOURCE_OTHER, WAGE_RATE_OF_PAY_FROM, WAGE_RATE_OF_PAY_TO,
                               WAGE_UNIT_OF_PAY, WORKSITE_POSTAL_CODE))

h1b <- subset(h1b, h1b$VISA_CLASS=="H1B")
h1b <- subset(h1b, h1b$EMPLOYER_COUNTRY=="UNITED STATES OF AMERICA")
h1b <- subset(h1b, select = -c(VISA_CLASS, EMPLOYER_COUNTRY))
h1b <- subset(h1b, h1b$TOTAL_WORKERS=='1')
h1b <- subset(h1b, select = -c(TOTAL_WORKERS))
h1b <- subset(h1b, h1b$CASE_STATUS != "WITHDRAWN")
h1b$CASE_STATUS[h1b$CASE_STATUS == "CERTIFIEDWITHDRAWN"] <- "CERTIFIED"

h1b <- h1b[!(h1b$FULL_TIME_POSITION==""),]
h1b <- h1b[!(h1b$PW_UNIT_OF_PAY==""),]
h1b <- h1b[!(h1b$H.1B_DEPENDENT==""),]
h1b <- h1b[!(h1b$WILLFUL_VIOLATOR==""),]
h1b <- droplevels(h1b)

pw_unit_to_yearly <- function(prevailing_wage, pw_unit_of_pay) {
  return(ifelse(pw_unit_of_pay == "Year",
                prevailing_wage,
                ifelse(pw_unit_of_pay == "Hour",
                       2080*prevailing_wage,
                       ifelse(pw_unit_of_pay== "Week",
                              52*prevailing_wage,
                              ifelse(pw_unit_of_pay == "Month",
                                     12*prevailing_wage,
                                     26*prevailing_wage)))))
}


h1b %>%
  filter(!is.na(PW_UNIT_OF_PAY)) %>%
  mutate(PREVAILING_WAGE = as.numeric(PREVAILING_WAGE)) %>%
  mutate(PREVAILING_WAGE = pw_unit_to_yearly(PREVAILING_WAGE, PW_UNIT_OF_PAY)) %>%
  select(everything()) -> h1bconvert

summary(h1bconvert$PREVAILING_WAGE)
boxplot(h1bconvert$PREVAILING_WAGE)

sh1b <- subset(h1bconvert,h1bconvert$PREVAILING_WAGE<10000000)
boxplot(sh1b$PREVAILING_WAGE)

ssh1b <- subset(h1bconvert,h1bconvert$PREVAILING_WAGE<1000000)
boxplot(ssh1b$PREVAILING_WAGE)

h1bconvert <- h1bconvert[!(h1bconvert$PREVAILING_WAGE<10000 | h1bconvert$PREVAILING_WAGE>400000),]
summary(h1bconvert$PREVAILING_WAGE)

h1bconvert <- subset(h1bconvert, select = -c(PW_UNIT_OF_PAY))
h1bclean <- h1bconvert

##dummy <- dummyVars( ~., data=h1bclean)
##dummy.h1b <- predict(dummy, newdata = h1bclean)

smote <- SMOTE(CASE_STATUS ~ ., data = h1bclean, perc.over = 3000, perc.under = 125)
table(smote$CASE_STATUS)
rose <- ROSE(CASE_STATUS ~ ., data = h1bclean)$data
table(rose$CASE_STATUS)

h1bsmote <- smote

ctrl <- trainControl(method="cv", number=5,
                     classProbs=TRUE,
                     #function used to measure performance
                     summaryFunction = twoClassSummary, #multiClassSummary for non binary
                     allowParallel =  TRUE)

trainIndex <- createDataPartition(h1bsmote$CASE_STATUS, p=.8, list=F) 
h1b.train <- h1bsmote[trainIndex,]
h1b.test <- h1bsmote[-trainIndex,]

set.seed(199)
h1b.log <- train(CASE_STATUS ~ ., data=h1b.train, method="glm", family="binomial", metric="ROC", trControl=ctrl)
varImp(h1b.log)
getTrainPerf(h1b.log)
h1b.log
p.log <- predict(h1b.log, h1b.test)
confusionMatrix(p.log, h1b.test$CASE_STATUS)

set.seed(199)
h1b.lda <-  train(CASE_STATUS ~ ., data=h1b.train, method="lda", family="binomial", metric="ROC", trControl=ctrl)
varImp(h1b.lda)
getTrainPerf(h1b.lda)
h1b.lda
p.lda <- predict(h1b.lda, h1b.test)
confusionMatrix(p.lda, h1b.test$CASE_STATUS)

set.seed(199)
h1b.rpart <- train(CASE_STATUS ~ ., data=h1b.train, trControl = ctrl, metric="ROC", method="rpart")
p.rpart <- predict(h1b.rpart, h1b.test)
confusionMatrix(p.rpart, h1b.test$CASE_STATUS)
rpart.plot(h1b.rpart$finalModel)

set.seed(199)
h1b.boost <-  train(CASE_STATUS ~ ., data=h1b.train, method="LogitBoost", family="binomial", metric="ROC", trControl=ctrl)
varImp(h1b.boost)
getTrainPerf(h1b.boost)
h1b.boost
p.boost <- predict(h1b.boost, h1b.test)
confusionMatrix(p.boost, h1b.test$CASE_STATUS)

rValues <- resamples(list(rpart=h1b.rpart, log=h1b.log, logboost=h1b.boost, lda=h1b.lda))
bwplot(rValues, metric="ROC")
bwplot(rValues, metric="Sens") #Sensitvity
bwplot(rValues, metric="Spec")
summary(rValues)


#NAIVE BAYES:
##Naive Bayes
modelLookup("nb") #we have some paramters to tune such as laplace correction
set.seed(192)
library(MLmetrics)
#h1B.nb <- train(CASE_STATUS ~ .-EMPLOYER_NAME - WORKSITE_POSTAL_CODE, data = h1BTrain,
#trControl = ctrl,
#metric = "ROC", #using AUC to find best performing parameters
#method = "nb")
h1B.nb <- train(CASE_STATUS ~ ., data = h1BTrain,
                trControl = ctrl,
                metric = "ROC", #using AUC to find best performing parameters
                method = "nb")
h1B.nb
getTrainPerf(h1B.nb)
varImp(h1B.nb)
plot(h1B.nb)
p.nb<- predict(h1B.nb,h1BTest)
confusionMatrix(p.nb, h1BTest$CASE_STATUS)
nb <- save(p.nb, file = "naivebayesh1b.rda")
load("naivebayesh1b.rda")

p.rpart2 <- predict(h1b.rpart, h1b.test, type="prob")
p.log2 <- predict(h1b.log, h1b.test, type="prob")
p.lda2 <- predict(h1b.lda, h1b.test, type="prob")
p.boost2 <- predict(h1b.boost, h1b.test, type="prob")

rpart.roc <- roc(h1b.test$CASE_STATUS, p.rpart2$DENIED)
log.roc <- roc(h1b.test$CASE_STATUS, p.log2$DENIED)
lda.roc <- roc(h1b.test$CASE_STATUS, p.lda2$DENIED)
logboost.roc <- roc(h1b.test$CASE_STATUS, p.boost2$DENIED)

plot(smooth(rpart.roc), col="black")
plot(smooth(log.roc), add=T, col="red")
plot(smooth(lda.roc), add=T, col="blue")
plot(smooth(logboost.roc), add=T, col="orange")
legend(x=.33, y=.3, cex=1, legend=c("rpart","logistic","lda","logboost"), col=c("black", "red", "blue", "orange"), lwd=3)
