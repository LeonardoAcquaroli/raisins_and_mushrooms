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
library(performance) # evaluating models
library(see)         # evaluating models
library(corrplot)# correlation plot 
library(GGally)  # correlation plot  
library(car)    
library(leaps)   # best subset
library(bestglm) # best subset for logistic
library(MASS)    # diagnostics
library(olsrr)   # diagnostics
library(influence.ME)
library(DHARMa)

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

cor_table<-cor(raisins) 

corrplot(cor_table, type = "upper",     #first corr plot
         tl.col = "black", tl.srt = 45)

ggcorr(raisins, method = c("everything", "pearson")) #heatmap plot

ggpairs(raisins_corr, columns = 1:6, ggplot2::aes(colour= Class_literal)) #cor by groups

########### OLS REGRESSION ####################
ols <- lm("Class ~ .",data=raisins)
summary(ols)
hist(fitted(ols))
#check heteroskedasticity
check_heteroscedasticity(ols) # there is heteroskedasticity
check_model(ols)
vif(ols) #uupsie
sqrt(vif(ols)) > 2 # uupsieee x2

# Deal with correlation by using best subset selection
regfit.full=regsubsets(Class ~.,data=raisins, nvmax=7)
reg.summary=summary(regfit.full)
plot(reg.summary$cp,xlab="Number of Variables",ylab="Cp")
which.min(reg.summary$cp)
plot(regfit.full,scale="Cp")

# run the OLS again

ols_imp <- lm("Class ~ Area + MajorAxisLength + 
              Eccentricity + ConvexArea + Perimeter",data=raisins)
summary(ols_imp)
vif(ols_imp) # the VIF is still too high!
sqrt(vif(ols_imp)) > 2

regfit.full=regsubsets(Class ~.,data=raisins, nvmax=3)
reg.summary=summary(regfit.full)
plot(regfit.full,scale="Cp")

ols_imp2 <- lm("Class ~ Eccentricity + Perimeter",data=raisins)
summary(ols_imp2)
vif(ols_imp2)
sqrt(vif(ols_imp2)) > 2

check_model(ols_imp2)
plot(ols_imp2)
check_heteroscedasticity(ols_imp2) # there is heteroskedasticity

# ols_imp2 is our final model as far as linear regressions goes. 
# let's apply further diagnostics

raw_res <- residuals(ols_imp2)
threshold <- 3 * sd(raw_res)  # Define threshold as 2 times the SD
threshold# how are the residuals distributed?
hist(raw_res, breaks = 30, main = "Histogram of Raw Residuals", xlab = "Raw Residuals")

#cook's distance
cooks_dist <- cooks.distance(ols_imp2)
cooks_threshold <- 4/897
plot(cooks_dist, raw_res, main = "Cook's Distance vs. Raw Residuals",
     xlab = "Cook's Distance", ylab = "Raw Residuals")
abline(h = 0, lty = 2, col = "red")  # Reference line at y = 0
abline(h = threshold, lty = 3, col = "blue")  # Threshold line for raw residuals (upper)
abline(h = -threshold, lty = 3, col = "blue")  # Threshold line for raw residuals (lower)
abline(v = cooks_threshold, lty = 3, col = "green")
text(cooks_dist, raw_res, labels = 1:length(cooks_dist), pos = 3)
# wow there are a lot of outliers

# most problematic observations appear to be: 695, 507, 837, 291, 86, 488
outliers_obs <- c(695, 507, 837, 291, 86, 488)
outliers_df <- raisins[outliers_obs, ]
non_outliers_df <- raisins[-outliers_obs, ]
p <- ggplot(non_outliers_df, aes(x = Perimeter, y = Eccentricity)) +
  geom_point() +
  labs(title = "Scatter Plot emphasising outliers",
       x = "Perimeter",
       y = "Eccentricity")

# Add outliers with distinct color and labels
p <- p +
  geom_point(data = outliers_df, color = "red", size = 3) +
  geom_text(data = outliers_df, aes(label = paste(rownames(outliers_df))), vjust = -1)

# Add reference lines for median values
p <- p +
  geom_vline(aes(xintercept = median(Perimeter)), linetype = "dashed") +
  geom_hline(aes(yintercept = median(Eccentricity)), linetype = "dashed")

# Plot the regression line with and without outliers
p <- p +
  geom_smooth(method = "lm", se = FALSE) +
  geom_smooth(data = non_outliers_df, method = "lm", se = FALSE, linetype = "dashed", color = "blue")

# Display the plot
print(p)

#computation of the accuracy on the full dataset
fitvols_fulld = predict(ols_imp2, raisins) # fulld identifies the full dataset
predols_fulld = ifelse(fitvols_fulld < 0.50, 0, 1)
accols_fulld = raisins$Class == predols_fulld
table(accols_fulld)
accuracy_ols_fulld <- 776/900
accuracy_ols_fulld

#ROBUST OLS
ols_robust <- lm_robust(Class ~ Eccentricity + Perimeter , data = raisins, se_type = "HC2")
summary(ols_robust)
check_model(ols_robust)

fitvolsrob_fulld = predict(ols_robust, raisins) # fulld identifies the full dataset
predolsrob_fulld = ifelse(fitvolsrob_fulld < 0.50, 0, 1)
accolsrob_fulld = raisins$Class == predolsrob_fulld
table(accolsrob_fulld)
accuracy_olsrob_fulld <- 776/900
accuracy_olsrob_fulld

# what if we remove the "outliers" previously mentioned? 695, 507, 837, 291, 86, 488
raisins_noout <- raisins[!(row.names(raisins) %in% c(695, 507, 837, 291, 86, 488)), ]

#let's implement the models on this modified data set
ols_imp2_n <- lm("Class ~ Eccentricity + Perimeter",data=raisins_noout) # _n identifies no outliers
summary(ols_imp2_n)
fitvols_fulld_n = predict(ols_imp2_n, raisins_noout) # fulld identifies the full dataset
predols_fulld_n = ifelse(fitvols_fulld_n < 0.50, 0, 1)
accols_fulld_n = raisins_noout$Class == predols_fulld_n
table(accols_fulld_n)
accuracy_ols_fulld_n <- 772/894
accuracy_ols_fulld_n

# same for the robust regression
ols_rob_n <- lm("Class ~ Eccentricity + Perimeter",data=raisins_noout) # _n identifies no-outliers
summary(ols_rob_n)
fitvolsrob_fulld_n = predict(ols_rob_n, raisins_noout) # fulld identifies the full dataset
predolsrob_fulld_n = ifelse(fitvolsrob_fulld_n < 0.50, 0, 1)
accolsrob_fulld_n = raisins_noout$Class == predolsrob_fulld_n
table(accolsrob_fulld_n)
accuracy_olsrob_fulld_n <- 772/894
accuracy_olsrob_fulld_n


# LOGISTIC REGRESSION
# Powered by https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4842399/

logistic_model <-  glm(Class ~ ., data = raisins, family = binomial(link = 'logit'))
tidy(logistic_model)
vif(logistic_model)
hist(fitted(logistic_model))


args(bestglm)
bestglm(raisins, family = gaussian, IC = "BIC")
imp_logistic_model <- glm(Class ~ Area + ConvexArea + Perimeter, 
                    data = raisins, 
                    family = binomial(link = 'logit'))
vif(imp_logistic_model)

# now for a minimal logistic with no vif issues

minimal1_logistic_model <-  glm(Class ~ Extent + Eccentricity, data = raisins, family = binomial(link = 'logit'))
vif(minimal1_logistic_model)

minimal2_logistic_model <-  glm(Class ~ Eccentricity + Perimeter, data = raisins,family = binomial(link = 'logit'))
vif(minimal2_logistic_model)


# create a table of the accuracies
accuracies <- data.frame(Model = character(),
                         Accuracy = numeric())

models <- list(logistic = logistic_model, 
               imp_logistic = imp_logistic_model,
               minimal_logit_extent = minimal1_logistic_model,
               minimal_logit_perimeter = minimal2_logistic_model,
               lpm = ols_imp2,
               lpm_rob = ols_robust)

for (model_name in names(models)) {
  model <- models[[model_name]]
  
  # Predict the outcome probabilities using the logistic regression model
  pred_pr <- predict(model, raisins, type = "response")
  # Convert the predicted probabilities to binary predictions (0 or 1)
  pred_val <- ifelse(pred_pr > 0.5, 1, 0)
  # Compare the predicted values with the actual outcome variable
  actual_values <- raisins$Class
  # Compute the number of correctly predicted values
  correctly_pred <- sum(pred_val == actual_values)
  # Compute the percentage of correctly predicted values
  accuracy <- correctly_pred / length(actual_values) * 100
  # Append the accuracy to the "accuracies" data frame
  accuracies <- rbind(accuracies, data.frame(Model = model_name, Accuracy = accuracy))
}

accuracies

# we take "minimal_logit_perimeter" as the best one. 
# we compute additional diagnostics

predicted_probs_logit <- predict(minimal2_logistic_model, type = "response")
residuals_logit <- raisins$Class - predicted_probs_logit

plot(predicted_probs_logit, raisins$Class, xlab = "Predicted Probabilities", ylab = "Observed Responses",
     main = "Observed vs. Predicted Probabilities", pch = 16)

#check relations with each predictor
plot(raisins$Eccentricity, residuals_logit, xlab = "Eccentricity", ylab = "Standardized Residuals",
     main = "Standardized Residuals vs. Predictor 1", pch = 16)
plot(raisins$Perimeter, residuals_logit, xlab = "Perimeter", ylab = "Standardized Residuals",
     main = "Standardized Residuals vs. Predictor 1", pch = 16)

# Diagnostics with DHARMa

# Create a simulated residuals object
simulated_residuals <- simulateResiduals(minimal2_logistic_model, n = 100)
# Plot standardized residuals using DHARMa's built-in diagnostic plots
plot(simulated_residuals)

# Compare all the models
compare_performance(ols_imp2, ols_robust, minimal2_logistic_model)
plot(compare_performance(ols_imp2, minimal2_logistic_model, rank = TRUE, verbose = FALSE))

accuracies

# 4. RIDGE # I didn't check this parts though
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

#### THINGS TO DO #####
# Add accuracy of LPM, Lasso, Ridge
# Add to performance comparison Lasso and Ridge
# Add to performance comparison plot Lasso and Ridge

