# RtistsFinalProject
data = read.csv('Life Expectancy Data.csv')
dim(data)

#Data cleaning
anyNA(data)
sum(is.na(data))
22*2938
2563/64636
data = na.omit(data)
anyNA(data)
dim(data)
2938-1649

summary(data)
str(data)
length(unique(data$Country)) #133 unique countries
data$Status = as.factor(data$Status)

library(dplyr)
data2 <- data %>% select(-c(Country,Year))

# Part 1: Regression model
#Model Selection and Regularization
#Best subset Selection (p<30)
#Cross Validation -> LOOCV?
#Regularization can yield better prediciton accuracy
#Check for multicollinearity, non constant variance, non linearity

#Part 2: Classification Model
