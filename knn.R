library(dplyr)
library(tidyverse)
library(caret)
library(ggplot2)
library(e1071)
library(caTools)
library(class)
library(pROC)


#Dataset
#Class = 1 if raisin is of Kecimen type, 0 if it is Besnis
raisins = read.csv(
  "https://raw.githubusercontent.com/LeonardoAcquaroli/raisins_and_mushrooms/main/datasets/Raisin_Dataset.csv",
  sep = ";"
)

raisins = raisins[,-8]
set.seed(42)
training_index = createDataPartition(raisins$Class, p=0.7, list = FALSE)
# index of the train set examples

train = raisins[training_index,]
test = raisins[-training_index,]

# Loading data
data(raisins)
head(raisins)

#Scaling
train.array <- scale(train[, 1:7])
test.array<- scale(test[, 1:7])

training_labels=train$Class

# KNN Model 

classifier_knn <- knn(train,
                      test ,
                      cl =training_labels,
                      k = 1)
classifier_knn

# Confusion Matrix

cm <- table(test$Class, classifier_knn)
cm

confusionMatrix(as.factor(classifier_knn),as.factor(test$Class))

# K = 3
classifier_knn <- knn(train,
                      test ,
                      cl =training_labels,
                      k = 3)
classifier_knn

# Confusion Matrix
cm <- table(test$Class, classifier_knn)
cm

confusionMatrix(as.factor(classifier_knn),as.factor(test$Class))

# K = 5
classifier_knn <- knn(train,
                      test ,
                      cl =training_labels,
                      k = 5)
classifier_knn

# Confusiin Matrix
cm <- table(test$Class, classifier_knn)
cm

confusionMatrix(as.factor(classifier_knn),as.factor(test$Class))
# K = 7
classifier_knn <- knn(train,
                      test ,
                      cl =training_labels,
                      k = 7)
classifier_knn

# Confusiin Matrix
cm <- table(test$Class, classifier_knn)
cm

confusionMatrix(as.factor(classifier_knn),as.factor(test$Class))

# K = 15
classifier_knn <- knn(train,
                      test ,
                      cl =training_labels,
                      k = 15)
classifier_knn

# Confusiin Matrix
cm <- table(test$Class, classifier_knn)
cm

confusionMatrix(as.factor(classifier_knn),as.factor(test$Class))
# K = 20
classifier_knn <- knn(train,
                      test ,
                      cl =training_labels,
                      k = 20)
classifier_knn

# Confusiin Matrix
cm <- table(test$Class, classifier_knn)
cm

confusionMatrix(as.factor(classifier_knn),as.factor(test$Class))

#range of K values 
k_values <- 1:50

# store results
accuracy_scores <- numeric(length(k_values))

# Iterate
for (i in seq_along(k_values)) {
  #KNN model
  classifier_knn <- knn(train, test, cl = training_labels, k = k_values[i])
  
  # Convert to factors
  classifier_knn <- as.factor(classifier_knn)
  actual_labels <- as.factor(test$Class)
  
  #Accuracy
  cm <- confusionMatrix(classifier_knn, actual_labels)
  accuracy_scores[i] <- cm$overall["Accuracy"]
}
  

# Best K value 
best_k <- k_values[which.max(accuracy_scores)]

print(paste("Accuracy scores:", accuracy_scores))
print(paste("Best K value:", best_k))


#Data frame with K values and accuracy
results <- data.frame(K = k_values, Accuracy = accuracy_scores)


# Plot
ggplot(results, aes(x = K, y = 1 - Accuracy)) +
  geom_line() +
  geom_point() +
  labs(x = "K", y = "Test Error") +
  ggtitle("Test Error vs K") +
  theme_minimal()




