library(MASS)
library(olsrr)
library(car)


raisins <- read.csv(
  "https://raw.githubusercontent.com/LeonardoAcquaroli/raisins_and_mushrooms/main/datasets/Raisin_Dataset.csv",
  sep = ";"
)

raisins <- raisins[,-8] # Class = 1 if raisin is of Kecimen type, 0 if it is Besni
ols <- lm("Class~.", data = raisins)
summary(ols)

# check multicollinearity, using VIF
vif(ols)
sqrt(vif(ols)) > 2


par(mfrow = c(2, 2))
plot(ols)


# Studentized Residuals Plot
ols_plot_resid_stud(ols)

ols_plot_resid_stand(ols)

# Deleted Studentized Residual vs Fitted Values Plot for detecting outliers
ols_plot_resid_stud_fit(ols)

# Studentized Residuals vs Leverage Plot for detecting influential observations
ols_plot_resid_lev(ols)


# Cook Distance : it is used to identify influential data points. It depends on both the residual and leverage i.e it takes it account both the x value and y value of the observation.
# Steps to compute Cook’s distance:

## delete observations one at a time.
## refit the regression model on remaining \((n−1)\) observations
## examine how much all of the fitted values change when the ith observation is deleted.
## A data point having a large cook’s d indicates that the data point strongly influences the fitted values.

par(mfrow = c(1,2))
plot(ols, 4)
plot(ols,5)


 
ols_plot_cooksd_bar(ols)

ols_plot_cooksd_chart(ols)




