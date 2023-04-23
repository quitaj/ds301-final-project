# RtistsFinalProject

data = read.csv('Life Expectancy Data.csv') 
dim(data)

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
data2 <- data %>% select(-c(Country,Year))
summary(data2)

# Part 1: Regression model

#Model Selection 
#Regularization can yield better prediciton accuracy 
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

train = data2[train_index,]
test = data2[-train_index,]

library(leaps) 
best.train = regsubsets(Life.expectancy~.,data=train,nbest=1,nvmax=19)

val.errors = rep(NA,19) 
for(i in 1:19){ 
  test.mat = model.matrix(Life.expectancy~.,data=test)
  coef.m = coef(best.train,id=i)
  pred = test.mat[,names(coef.m)]%*%coef.m 
  val.errors[i] = mean((test$Life.expectancy-pred)^2) 
  } 

plot(val.errors)
which.min(val.errors) 

coef(best.train,9)
coef(best.train,12)

pred12 = test.mat[,names(coef(best.train,id=12))]%*%coef(best.train,id=12)
pred12[1]

#TEST MSE = 13.5
val.errors[12]
sqrt(val.errors[12])
best.train.sum = summary(best.train)

p = rowSums(best.train.sum$which) #number of predictors + intercept in the model 
rss = best.train.sum$rss
adjr2 = best.train.sum$adjr2
cp = best.train.sum$cp
AIC = n*log(rss/n) + 2*(p)
BIC = n*log(rss/n) + (p)*log(n)
cbind(p,AIC,BIC,adjr2,cp,rss)

which.min(AIC)
which.min(BIC)
which.max(adjr2)
which.min(cp)

fit = lm(Life.expectancy ~ Status + Adult.Mortality + infant.deaths + Alcohol + percentage.expenditure + BMI + under.five.deaths + Diphtheria + HIV.AIDS + thinness.5.9.years + Income.composition.of.resources + Schooling ,train)
summary(fit)

library(car)
vif(fit)
#VIF for infant.deaths and under.five.deaths are very high. infant.deaths is removed to resolve multicollinearity

fit2 = lm(Life.expectancy ~ Status + Adult.Mortality + Alcohol + percentage.expenditure + BMI + under.five.deaths + Diphtheria + HIV.AIDS + thinness.5.9.years + Income.composition.of.resources + Schooling, train)
vif(fit2)

par(mfrow=c(2,2))
plot(fit2)
#residuals VS fitted values fairly random -> constant variance
#scale-location plot also fairly randoma and linear


# Part 2: Classification Model

