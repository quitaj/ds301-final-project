# RtistsFinalProject

data = read.csv('Life Expectancy Data.csv') 
dim(data)
summary(data)

#Data cleaning 
anyNA(data) 
sum(is.na(data)) 
data = na.omit(data) 
anyNA(data) 
dim(data) 
2938-1649

summary(data) 
str(data) 
length(unique(data$Country)) 
#133 unique countries 
data$Status = as.factor(data$Status)

library(dplyr) 
data2 <- data %>% select(-c(Country))
summary(data2)

# Part 1: Regression model

#Model Selection 
#Check for multicollinearity, non constant variance, non linearity

#REGULAR MLR MODEL
#Significant F Statistic
model = lm(Life.expectancy ~ .,data=data2)
summary(model)

#BEST SUBSET SELECTION WITH CV
n = dim(data2)[1]

#DATA SPLIT
set.seed(10) 
train_index = sample(1:n,n/2,rep=FALSE)

train = (data2$Year == 2014)
test = (data2$Year == 2015)

library(leaps) 
best.train = regsubsets(Life.expectancy~.,data=train,nbest=1,nvmax=19)

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
which.min(AIC)
which.min(BIC)
which.max(adjr2)
which.min(cp)

coef(best.train,9)

pred9 = test.mat[,names(coef(best.train,id=9))]%*%coef(best.train,id=9)
pred9[1]

#TEST MSE = 13.65
val.errors[9]
sqrt(val.errors[9])

fit = lm(Life.expectancy ~ Adult.Mortality + infant.deaths + percentage.expenditure + BMI + under.five.deaths + Diphtheria + HIV.AIDS +  Income.composition.of.resources + Schooling ,train)
summary(fit)

library(car)
vif(fit)
#VIF for infant.deaths and under.five.deaths are very high. infant.deaths is removed to resolve multicollinearity

fit2 = lm(Life.expectancy ~ Adult.Mortality + percentage.expenditure + BMI + under.five.deaths + Diphtheria + HIV.AIDS +  Income.composition.of.resources + Schooling, train)
summary(fit2)
vif(fit2)

par(mfrow=c(2,2))
plot(fit2)
#residuals VS fitted values fairly random -> constant variance
#scale-location plot also fairly randoma and linear

pred = predict(fit2,newdata = test)
MSE_test = mean((test$Life.expectancy - pred)^2)
MSE_test

# Part 2: Classification Model

