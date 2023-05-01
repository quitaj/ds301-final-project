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

#test MSE associated with the model is 9.993828
ridge.pred = predict(ridge.train,s=bestlambda,newx=x[test,])
mean((ridge.pred-Y.test)^2)

coef(ridge_model)
dim(coef(ridge_model)) 


#trying the data on lasso, gives me a test MSE of 9.555846.

cv.out.lasso = cv.glmnet(x[train,],Y[train],alpha = 1, lambda = grid) 
plot(cv.out.lasso)
bestlambda2 = cv.out.lasso$lambda.min
bestlambda2

lasso.train = glmnet(x[train,],Y[train],alpha=1,lambda=grid)

lasso.pred = predict(lasso.train,s=bestlambda2,newx=x[test,])
mean((lasso.pred-Y.test)^2)

final.lasso = glmnet(x,Y,alpha=1,lambda=bestlambda2)
coef(final.lasso)


#Trying lasso with a different lambda grid, gives test mse of 9.556944.
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

#Final Model

final = lm(Life.expectancy ~ Adult.Mortality + HIV.AIDS +  Income.composition.of.resources + Total.expenditure + Status + percentage.expenditure + under.five.deaths + Diphtheria + thinness..1.19.years, data)
summary(final)

par(mfrow=c(2,2))
plot(final)

# Part 2: Classification Model

### Resetting Data
set.seed(1)
data = read.csv('Life Expectancy Data.csv') 
data1 <- data %>% filter(data$Year == 2013|data$Year == 2014)

data1$Status = as.factor(data1$Status)
data1$Country = as.factor(data1$Country)
data1 <- data1 %>% select(-c("GDP","Population"))
data = na.omit(data1)
train = data%>% filter(Year == 2013)
test = data%>% filter(Year == 2014)

test <- subset(test, select = -c(1, 2))
train <- subset(train, select = -c(1, 2))
data <- subset(data, select = -c(1, 2))



## Logistic Regression
library(MASS)
library(leaps) 

### Best Subset Selection w/ Cross Validation
best.train = regsubsets(Status~.,data=train,nbest=1,nvmax=17)

val.errors = rep(NA,17) 
for(i in 1:17){ 
  test.mat = model.matrix(Status~.,data=test)
  coef.m = coef(best.train,id=i)
  pred = test.mat[,names(coef.m)]%*%coef.m 
  val.errors[i] = mean((test$Status-pred)^2) 
} 

summary(best.train)

best.train.sum = summary(best.train)
n = n = dim(train)[1]
p = rowSums(best.train.sum$which) #number of predictors + intercept in the model 
rss = best.train.sum$rss
AIC = n*log(rss/n) + 2*(p)
plot(AIC,type='b')

coef(best.train,4)

### Fit Model

glm.fit = glm(Status~ Life.expectancy + Alcohol + BMI + Total.expenditure , data=train, family='binomial')

summary(glm.fit)
head(glm.fit$fitted.values)
head(data)

glm.prob = predict(glm.fit,test,type='response') 

### Matrix with .5 threshold and new at .3

glm.pred = rep('Developed',131)
glm.pred[glm.prob > 0.5] ='Developing'
table(glm.pred,test$Status)
glm.pred = as.factor(glm.pred)
mean(test$Status!=glm.pred)

glm.pred = rep('Developed',131)
glm.pred[glm.prob > 0.3] ='Developing'
table(glm.pred,test$Status)
glm.pred = as.factor(glm.pred)
mean(test$Status!=glm.pred)

## LDA
library(ggplot2)
library(MASS)

### Showing semi-normal distribution and small data set

hist(data$Life.expectancy, main = "Life Expectancy")
hist(data$Alcohol, main = "Alcohol")
hist(data$BMI, main = "BMI")
hist(data$Total.expenditure, main = "Total.expenditure")

head(test$Status)

### Fit Model

lda.fit = lda(Status~Life.expectancy + Alcohol + BMI + Total.expenditure,data= train)

lda.fit

lda.pred = predict(lda.fit ,test)

### Matrix with .5 threshold and new at .3

table(lda.pred$class,test$Status)
mean(lda.pred$class!=test$Status)

lda.class = rep('Developed', 131)
lda.class[lda.pred$posterior[,2]>=0.3] = 'Developing'
lda.class = as.factor(lda.class)
table(lda.class,test$Status)
mean(lda.class!=test$Status) 



## QDA 

### Shows that variance is not the same - justifies why we used QDA after LDA

var(data$Life.expectancy)
var(data$Alcohol)
var(data$BMI)
var(data$Total.expenditure)

### Fit the Model

qda.fit = qda(Status ~ Life.expectancy + Alcohol + BMI + Total.expenditure,data = data)

### Matrix with .5 threshold and new at .3

qda.pred = predict(qda.fit,test)
table(qda.pred$class,test$Status)
mean(qda.pred$class!=test$Status)

qda.class = rep('Developed', 131)
qda.class[qda.pred$posterior[,2]>=0.3] = 'Developing'
qda.class = as.factor(qda.class)
table(qda.class,test$Status)
mean(qda.class!=test$Status) 


## KNN

library(class)

### Factor and standardize the data set

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

### K - Selection Using 5 k-fold CV

library(caret)

tests = 1:163

data2 = data[-tests,]
standardized.X2 = scale(data2[,-18])

flds <- createFolds(data2$Status, k = 5, list = TRUE, returnTrain = FALSE)
names(flds)

K= c(1,3,5,7,9,11,13,15,17,19,21)

cv_error = matrix(NA, 5, 11)
head(cv_error)

for(j in 1:11){
  k = K[j]
  for(i in 1:5){
    test_index = flds[[i]]
    testX = standardized.X2[test_index,]
    trainX = standardized.X2[-test_index,]
    
    trainY = data2$Status[-test_index]
    testY = data2$Status[test_index]
    
    knn.pred = knn(trainX,testX,trainY,k=k)
    cv_error[i,j] = mean(testY!=knn.pred)
  }
}

print(cv_error)
apply(cv_error,2,mean)

### Matrix of results - using best k

knn.pred = knn(train.X,test.X,train.Y,k=1)
head(knn.pred)

table(knn.pred,test.Y)
mean(test.Y!=knn.pred)


