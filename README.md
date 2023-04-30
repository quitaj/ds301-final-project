# RtistsFinalProject
library(dplyr)
data = read.csv('Life Expectancy Data.csv') 
data1 <- data %>% filter(data$Year == 2013|data$Year == 2014)
dim(data1)
summary(data1)
sum(is.na(data1)) 
# GDP and Population have the highest number of NA values, we decided to drop them 

str(data1) 
length(unique(data1$Country)) 
#183 unique countries 
data1$Status = as.factor(data1$Status)
data1$Country = as.factor(data1$Country)
data1 <- data1 %>% select(-c("GDP","Population"))
summary(data1)

#Data cleaning 
sum(is.na(data1)) 
data = na.omit(data1) #removed 50-60 observations from train and test set
anyNA(data) 
dim(data) 
376-325

#DATA SPLIT
set.seed(1)
train = data%>% filter(Year == 2013)
dim(train)
summary(train)
test = data%>% filter(Year == 2014)
dim(test)
summary(test)
hist(data$Life.expectancy, main = "Life Expectancy", xlab = "Life Expectancy") 
hist(train$Life.expectancy, main = "Train Life Expectancy", xlab = "Life Expectancy")
hist(test$Life.expectancy, main = "Test Life Expectancy", xlab = "Life Expectancy")

# Part 1: Prediction model

#Model Selection 
#Check for multicollinearity, non constant variance, non linearity

#BEST SUBSET SELECTION WITH CV
n = dim(data)[1]

#DATA SPLIT
library(leaps) 
best.train = regsubsets(Life.expectancy~.-Country-Year,data=train,nbest=1,nvmax=17)

val.errors = rep(NA,17) 
for(i in 1:17){ 
  test.mat = model.matrix(Life.expectancy~.,data=test)
  coef.m = coef(best.train,id=i)
  pred = test.mat[,names(coef.m)]%*%coef.m 
  val.errors[i] = mean((test$Life.expectancy-pred)^2) 
  } 

plot(val.errors,xlab = "Number of Predictors", ylab = "Test MSE")
which.min(val.errors)

coef(best.train,14)

#TEST MSE = 9.313424
val.errors[14]
sqrt(val.errors[14])

fit = lm(Life.expectancy ~ Adult.Mortality + HIV.AIDS +  Income.composition.of.resources + Total.expenditure + Status + infant.deaths + Alcohol + percentage.expenditure + Hepatitis.B + under.five.deaths + Diphtheria + thinness..1.19.years + thinness.5.9.years + Schooling, data)
summary(fit)

par(mfrow=c(2,2))
plot(fit)
#residuals VS fitted values fairly random -> constant variance
#scale-location plot also fairly randoma and linear

#Shrinkage Methods
library(glmnet)
set.seed(1)
x = model.matrix(Life.expectancy~.-Country-Year,data=data)[,-1] 
Y = data$Life.expectancy

grid = 10^seq(10,-2,length=100)
ridge_model = glmnet(x,Y,alpha=0, lambda=grid)
set.seed(1)

rownames(data) <- 1:nrow(data)
train = as.integer(rownames(data[data$Year==2013,]))
test = as.integer(rownames(data[data$Year==2014,]))

Y.test = Y[test]

ridge.train = glmnet(x[train,],Y[train],alpha=0,lambda=grid)

cv.out = cv.glmnet(x[train,],Y[train],alpha = 0, lambda = grid) 
plot(cv.out)
bestlambda = cv.out$lambda.min
bestlambda

#test MSE associated with the model is 12.392
ridge.pred = predict(ridge.train,s=bestlambda,newx=x[test,])
mean((ridge.pred-Y.test)^2)

coef(ridge_model)
dim(coef(ridge_model)) 


#trying the data on lasso, gives me a test MSE of 12.191.

cv.out.lasso = cv.glmnet(x[train,],Y[train],alpha = 1, lambda = grid) 
plot(cv.out.lasso)
bestlambda2 = cv.out.lasso$lambda.min
bestlambda2

lasso.train = glmnet(x[train,],Y[train],alpha=1,lambda=grid)

lasso.pred = predict(lasso.train,s=bestlambda2,newx=x[test,])
mean((lasso.pred-Y.test)^2)

final.lasso = glmnet(x,Y,alpha=1,lambda=bestlambda2)
coef(final.lasso)


#Trying lasso with a different lambda grid, gives test mse of 11.8175.
grid = 5^seq(5,-2,length=50)

cv.out.lasso = cv.glmnet(x[train,],Y[train],alpha = 1, lambda = grid) 
plot(cv.out.lasso)
bestlambda2 = cv.out.lasso$lambda.min
bestlambda2

lasso.train = glmnet(x[train,],Y[train],alpha=1,lambda=grid)

lasso.pred = predict(lasso.train,s=bestlambda2,newx=x[test,])
mean((lasso.pred-Y.test)^2)

final.lasso = glmnet(x,Y,alpha=1,lambda=bestlambda2)
coef(final.lasso)

#using the tree thing to see which predictors may be more important 

library(tree)
tree.data = tree(Life.expectancy~.-Country,data=data, subset=train)

summary(tree.data)
plot(tree.data)
text(tree.data,pretty=0)

# Part 2: Classification Model

test <- subset(test, select = -c(1, 2))
train <- subset(train, select = -c(1, 2))
data <- subset(data, select = -c(1, 2))

### LDA
library(MASS)

lda.fit = lda(Status~.,data= train)

lda.fit

lda.pred = predict(lda.fit ,test)
names(lda.pred)

head(lda.pred$class)
head(lda.pred$posterior)

table(lda.pred$class,test$Status)
mean(lda.pred$class!=test$Status) 
mean(lda.pred$class==test$Status)

### QDA

qda.fit = qda(Status ~ .,data = train)

qda.pred = predict(qda.fit,test)

table(qda.pred$class,test$Status)

mean(qda.pred$class==test$Status)

### KNN

library(ISLR2)

data$Status <- as.numeric(as.factor(data$Status))
test$Status <- as.numeric(as.factor(test$Status))
train$Status <- as.numeric(as.factor(train$Status))

standardized.data = scale(data[,-20])
standardized.test = scale(test[,-20])
standardized.train = scale(train[,-20])

train.X = standardized.train
test.X = standardized.test
train.Y = train$Status
test.Y = test$Status

set.seed(1)
library(class)
knn.pred = knn(train.X,test.X,train.Y,k=1)
head(knn.pred)

table(knn.pred,test.Y)
mean(test.Y!=knn.pred)

###### How to choose K? ##########
###### Cross-validation ##########




