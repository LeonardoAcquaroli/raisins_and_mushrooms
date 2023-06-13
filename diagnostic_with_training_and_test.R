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
library(performance)
library(see)
library(corrplot)
library(GGally)
library(car) 
library(leaps)


# load the dataset
# Class = 1 if raisin is of Kecimen type, 0 if it is Besni
raisins <- read.csv(
  "https://raw.githubusercontent.com/LeonardoAcquaroli/raisins_and_mushrooms/main/datasets/Raisin_Dataset.csv",
  sep = ";"
)



# Here I'll do the correlations plots as well

# remove the column of the literal class
raisins_corr <- raisins
raisins <- raisins[,-8] # Class = 1 if raisin is of Kecimen type, 0 if it is Besni
head(raisins)



set.seed(2)
sample <- sample(c(TRUE, FALSE), nrow(raisins), replace=TRUE, prob=c(0.7,0.3))
train  <- raisins[sample, ]
test   <- raisins[!sample, ]





cor_table<-cor(raisins) 

corrplot(cor_table, type = "upper",     #first corr plot
         tl.col = "black", tl.srt = 45)

ggcorr(raisins, method = c("everything", "pearson")) #heatmap plot

ggpairs(raisins_corr, columns = 1:6, ggplot2::aes(colour= Class_literal, alpha = 0.000001)) #cor by groups

########### OLS REGRESSION ####################
ols <- lm("Class ~ .",data=train)
ols2 <-  lm("Class ~ Eccentricity + Perimeter", data = train)
summary(ols2)
vif(ols2)
summary(ols)
hist(fitted(ols))
#check heteroskedasticity
check_heteroscedasticity(ols) # there is heteroskedasticity
check_model(ols)
vif(ols) #uupsie
sqrt(vif(ols)) > 2 # uupsieee x2

# Deal with correlation by using best subset selection
regfit.full=regsubsets(Class ~.,data=train, nvmax=7)
reg.summary=summary(regfit.full)
names(reg.summary) 
plot(reg.summary$cp,xlab="Number of Variables",ylab="Cp")
which.min(reg.summary$cp)
plot(regfit.full,scale="Cp")

# run the OLS again

ols_imp <- lm("Class ~ Area + MajorAxisLength + 
              Eccentricity + ConvexArea + Perimeter",data=raisins)
summary(ols_imp)
vif(ols_imp) # the VIF is still too high!
sqrt(vif(ols_imp)) > 2

regfit.full=regsubsets(Class ~.,data=raisins, nvmax=7)
reg.summary=summary(regfit.full)
plot(regfit.full,scale="Cp")

ols_imp2 <- lm("Class ~ Eccentricity + Perimeter",data=train)
summary(ols_imp2)
vif(ols_imp2) # the VIF is still too high!
sqrt(vif(ols_imp2)) > 2

check_model(ols_imp2)
plot(ols_imp2)

# ols test set


fitvols = predict(ols2, test)
fitvols


predols = ifelse(fitvols < 0.50, 0, 1)
predols 

accols = test$Class == predols
table(accols)





#ROBUST OLS
ols_robust <- lm_robust(Class ~ Eccentricity + Perimeter , data = train, se_type = "HC2")
summary(ols_robust)
check_model(ols_robust)


fitvrob = predict(ols_robust, test)
fitvrob


predrob = ifelse(fitvrob < 0.50, 0, 1)
predrob 

accrob = test$Class == predrob
table(accrob) 













# LOGISTIC REGRESSION
logistic <-  glm(Class ~ ., data = train, family = binomial(link = 'logit'))
tidy(logistic)
vif(logistic)
hist(fitted(logistic))
check_model(logistic)




fitvlog = predict(logistic, test)
fitvlog


predlog = ifelse(fitvlog < 0.50, 0, 1)
predlog 

acclog = test$Class == predlog
table(acclog) 






# 4. RIDGE
x = model.matrix(Class~.-1, data = raisins)
x
y=raisins$Class
fit.ridge=glmnet(x,y,alpha=0)
fit.ridge$beta
plot(fit.ridge,xvar="lambda", label = TRUE)
cv.ridge=cv.glmnet(x,y,alpha=0)
plot(cv.ridge)
coef(cv.ridge)
mse(fitted(fit.ridge), raisins, "Class")





# 5. LASSO
fit.lasso=glmnet(x,y)
plot(fit.lasso,xvar="lambda",label=TRUE)
cv.lasso=cv.glmnet(x,y)
plot(cv.lasso)
coef(cv.lasso)
mse(fit.lasso, raisins, "Class")
predict(fit.lasso,newx = x)

print("Hello")

#### THINGS TO DO #####
# 1 Fix multicollinearity issues | RIDGE or PCE
# 2 use the compare_performance() function to compare our models | After fixing multicollinearity
# 3 visualization of modeel performance through 
#   plot(compare_performance(ols, robust_ols, logistic, ridge, lasso, rank = TRUE, verbose = FALSE))
