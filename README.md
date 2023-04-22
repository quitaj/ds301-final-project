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
#CV with Best subset Selection (p\<30) 
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

coef(best.train,12)

pred12 = test.mat[,names(coef(best.train,id=12))]%*%coef(best.train,id=12)
pred12[1]

#TEST MSE = 13.5
val.errors[12]
sqrt(val.errors[12])

compare = data.frame(pred12,test$Life.expectancy)
compare$difference = compare$test.Life.expectancy - compare$pred12
compare$absdiff = abs(compare$difference)
which.max(compare$absdiff)
mean(compare$absdiff)
compare[679,]

#StatusDeveloping
X1 = 1
#Adult.Mortality
X2 = 271
#infant.deaths
X3 = 64
#Alcohol
X4 = 0.01
#percentage.expenditure
X5 = 73.523582
#BMI
X6 = 18.6
#under.five.deaths
X7 = 86
#Diphtheria
X8 = 62
#HIV.AIDS
X9 = 0.1
#thinness.5.9.years
X10 = 17.5
#Income.composition.of.resources
X11 = 0.476
#Schooling
X12 = 10.0

Life.Expect = 54.6959176777 - 0.8062397040*X1 - 0.0168080414*X2 + 0.0810582649*X3 - 0.0715568562*X4 + 0.0005057659*X5 + 0.0238329322*X6 - 0.0620981675*X7 + 0.0207481648*X8 - 0.4324345947*X9 - 0.0440166584*X10 + 10.5942729760*X11 + 0.8310133851*X12
Life.Expect

par(mfrow=c(1,2))
plot(Life.Expect, which=3)

# Part 2: Classification Model

