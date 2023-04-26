# RtistsFinalProject
library(dplyr)
data = read.csv('Life Expectancy Data.csv') 
data1 <- data %>% filter(data$Year == 2013|data$Year == 2014)
dim(data1)
summary(data1)

str(data1) 
length(unique(data1$Country)) 
#183 unique countries 
data1$Status = as.factor(data1$Status)
data1$Country = as.factor(data1$Country)

#Data cleaning 
sum(is.na(data1)) 
data = na.omit(data1) #removed 50-60 observations from train and test set
anyNA(data) 
dim(data) 

#DATA SPLIT
train = data%>% filter(Year == 2013)
dim(train)
summary(train)
test = data%>% filter(Year == 2014)
dim(test)
summary(test)

# Part 1: Regression model

#Model Selection 
#Check for multicollinearity, non constant variance, non linearity

#REGULAR MLR MODEL
#Significant F Statistic
model = lm(Life.expectancy ~ .,data=data)
summary(model)

#BEST SUBSET SELECTION WITH CV
n = dim(data)[1]

#DATA SPLIT
library(leaps) 
best.train = regsubsets(Life.expectancy~.-Country-Year,data=train,nbest=1,nvmax=19)

val.errors = rep(NA,19) 
for(i in 1:19){ 
  test.mat = model.matrix(Life.expectancy~.,data=test)
  coef.m = coef(best.train,id=i)
  pred = test.mat[,names(coef.m)]%*%coef.m 
  val.errors[i] = mean((test$Life.expectancy-pred)^2) 
  } 

best.train.sum = summary(best.train)
p = rowSums(best.train.sum$which) #number of predictors + intercept in the model 
rss = best.train.sum$rss
adjr2 = best.train.sum$adjr2
cp = best.train.sum$cp
AIC = n*log(rss/n) + 2*(p)
BIC = n*log(rss/n) + (p)*log(n)
cbind(p,rss,adjr2,AIC,BIC,cp, val.errors)

plot(val.errors)
which.min(val.errors)

coef(best.train,4)

pred4 = test.mat[,names(coef(best.train,id=4))]%*%coef(best.train,id=4)
pred4[1]

#TEST MSE = 13.65
val.errors[4]
sqrt(val.errors[4])

fit = lm(Life.expectancy ~ Adult.Mortality + HIV.AIDS +  Income.composition.of.resources + Total.expenditure, data)
summary(fit)

par(mfrow=c(2,2))
plot(fit)
#residuals VS fitted values fairly random -> constant variance
#scale-location plot also fairly randoma and linear

# Part 2: Classification Model

