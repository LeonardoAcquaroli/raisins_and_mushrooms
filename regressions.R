library(dplyr)
library(tidyverse)
library(tidymodels)
library(glmnet)
library(estimatr)
library(stats)
library(tree)
library(maxLik)
library(Matrix)
library(caret)

# load the dataset
# Class = 1 if raisin is of Kecimen type, 0 if it is Besni
raisins = read.csv(
  "https://raw.githubusercontent.com/LeonardoAcquaroli/raisins_and_mushrooms/main/datasets/Raisin_Dataset.csv",
  sep = ";"
)
# remove the column of the literal class
#raisins = raisins %>% select(-Class_literal) # don't know why it doesn't work
raisins = raisins[,-8]
set.seed(42)
training_index = createDataPartition(raisins$Class, p=0.7, list = FALSE) # index of the train set examples
train = raisins[training_index,]
test = raisins[-training_index,]

#raisins$Class = as.factor(raisins$Class)
#raisins$Area = raisins$Area/1000 # scale Area by 1000 for better interpretability. Recall Area is the n. of pixel inside the boundaries of the raisin

# mse function
mse = function(predictions,data,y){
  residuals = (predictions - (data[c(y)]))
  mse = (1/nrow(data))*sum((residuals^2))
  return(mse)
}

# 1. ols
ols = lm("Class ~ .",data=train)
summary(ols)
ols_test_predictions = predict.lm(ols,newdata = test)
hist(fitted(ols))
mse(fitted(ols), train, "Class") #training error
mse(ols_test_predictions,test,"Class") #test error

# 2. robust ols
ols_robust = lm_robust(Class ~ ., data = train, se_type = "HC2")
summary(ols_robust)
## Robust ols with rlm by MASS
#summary(rlm(Class ~ ., raisins))
ols_robust_test_predictions = predict(ols_robust, newdata = test)
mse(fitted(ols_robust), train, "Class") #training error
mse(ols_robust_test_predictions, test, "Class") #test error

# 3. Logistic
logistic = glm(Class ~ ., data = train, family = binomial(link = 'logit'))
tidy(logistic)
hist(fitted(logistic))
#logistic by hand
logistic_test_predictions = predict(logistic, newdata = test)
mse(fitted(logistic), train, "Class")
mse(logistic_test_predictions, test, "Class")

# 4. Ridge
x = model.matrix(Class~.-1, data = train)
x
y=train$Class

fit.ridge=glmnet(x,y,alpha=0)
fit.ridge$beta
plot(fit.ridge,xvar="lambda", label = TRUE)
cv.ridge=cv.glmnet(x,y,alpha=0)
plot(cv.ridge)
coef(cv.ridge)
mse(fitted(fit.ridge), train, "Class")

# 5. Lasso
fit.lasso=glmnet(x,y)
plot(fit.lasso,xvar="lambda",label=TRUE)
cv.lasso=cv.glmnet(x,y)
plot(cv.lasso)
coef(cv.lasso)
mse(fit.lasso, raisins, "Class")
predict(fit.lasso,newx = x)

# 6. Tree
tree = tree(Class ~ ., data = raisins)
plot(tree)
text(tree)
#tree_test_predictions = predict(tree, newdata = test, type = "tree")
#tree_predictions = predict(tree, newdata = train[,-8], type = "tree")
#
#mse(tree_test_predictions, test, "Class")
#mse(tree_test_predictions, test, "Class")
