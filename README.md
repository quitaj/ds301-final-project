---
editor_options: 
  markdown: 
    wrap: 72
---

# RtistsFinalProject

```{r}
data = read.csv('Life Expectancy Data.csv') dim(data)

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
```

# Part 1: Regression model

#Model Selection #Best subset Selection (p\<30) #Cross Validation -\>
LOOCV? #Regularization can yield better prediciton accuracy #Check for
multicollinearity, non constant variance, non linearity

```{r}
n = dim(data)[1]

set.seed(10) 
train_index = sample(1:n,n/2,rep=FALSE)

train = data[train_index,]
test = data[-train_index,]

library(leaps) 
best.train = regsubsets(Life.expectancy~.,data=train,nbest=1,nvmax=20)

val.errors = rep(NA,20) 
for(i in 1:20){ 
  test.mat = model.matrix(Life.expectancy~.,data=test)
  coef.m = coef(best.train,id=i)
  pred = test.mat[,names(coef.m)]%*%coef.m 
  val.errors[i] = mean((test$Life.expectancy-pred)^2) 
  } 
which.min(val.errors) 
coef(best.train,11)
```

# Part 2: Classification Model
