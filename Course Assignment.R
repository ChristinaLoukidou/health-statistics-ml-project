
test <- read.csv("c:/Users/gzoid/OneDrive/Έγγραφα/Big Data Analytics/Εργασια/3. data/test.csv",stringsAsFactors = TRUE)
View(test)

train <- read.csv("c:/Users/gzoid/OneDrive/Έγγραφα/Big Data Analytics/Εργασια/3. data/train.csv",stringsAsFactors = TRUE)
View(train)



str(train)
str(test)

summary(train)



#Question1
library(rpart) 
library(rpart.plot)



trainq1data <- train[,c("weight", "SOFA", "wbc_min", "wbc_max","mech_vent")]
testq1data <- test[,c("weight", "SOFA", "wbc_min", "wbc_max","mech_vent")]


minsplit_values = c(10,20,30)

for(i in minsplit_values) {
  model<- rpart(mech_vent ~ weight + SOFA + wbc_min + wbc_max , method = "class", data = trainq1data, minsplit=i)
 
   pred <- predict(model, testq1data[,1:4] ,type = "class")
   
   
  cm = as.matrix(table(Actual = testq1data[,5], Predicted = pred))
  accuracy = sum(diag(cm)) / sum(cm) 
  precision = diag(cm) / colSums(cm) 
  recall = diag(cm) / rowSums(cm) 
  f1 = 2 * precision * recall / (precision + recall) 
  
  cat("\n minsplit =", i, "\n")
  cat("Accuracy:", accuracy, "\n")
  print(data.frame(precision, recall, f1))
  
  
  }




finalmodel <- rpart(mech_vent ~ weight + wbc_min + wbc_max + SOFA, method = "class", data = trainq1data, minsplit=20)
rpart.plot(finalmodel, extra = 104, nn = TRUE)


pred <- predict(finalmodel, testq1data[,1:4] ,type = "class")

table(train$mech_vent)
prop.table(table(train$mech_vent))

cm = as.matrix(table(Actual = testq1data[,5], Predicted = pred))
accuracy = sum(diag(cm)) / sum(cm) 
precision = diag(cm) / colSums(cm) 
recall = diag(cm) / rowSums(cm) 
f1 = 2 * precision * recall / (precision + recall) 
data.frame(precision, recall, f1)


#Question2


library(e1071)
library(MLmetrics)


trainq2_nocrdvs <- train[,c(13, 14, 15, 16, 18)]
testq2_nocrdvs <- test[,c(13, 14, 15, 16, 18)]
trainq2_crdvs <- train[,c(13, 14, 15, 16, 17, 18)]
testq2_crdvs <- test[,c(13, 14, 15, 16, 17, 18)]


model1 <- naiveBayes(sepsis ~ ., data = trainq2_nocrdvs)

pred1 <- predict(model1, testq2_nocrdvs[,-ncol(testq2_nocrdvs)])
actual1 <- testq2_nocrdvs$sepsis

ConfusionMatrix(pred1, actual1)

cat("Accuracy:", Accuracy(pred1, actual1))
cat("Precision (NO):", Precision(actual1, pred1, positive = "NO"))
cat("Precision (YES):", Precision(actual1, pred1, positive = "YES"))
cat("Recall (NO):", Recall(actual1, pred1, positive = "NO"))
cat("Recall (YES):", Recall(actual1, pred1, positive = "YES"))
cat("F1 (NO):", F1_Score(actual1, pred1, positive = "NO"))
cat("F1 (YES):", F1_Score(actual1, pred1, positive = "YES"))


model2 <- naiveBayes(sepsis ~ ., data = trainq2_crdvs)

pred2 <- predict(model2, testq2_crdvs[,-ncol(testq2_crdvs)])
actual2 <- testq2_crdvs$sepsis

ConfusionMatrix(pred2, actual2)

cat("Accuracy:", Accuracy(pred2, actual2))
cat("Precision (NO):", Precision(actual2, pred2, positive = "NO"))
cat("Precision (YES):", Precision(actual2, pred2, positive = "YES"))
cat("Recall (NO):", Recall(actual2, pred2, positive = "NO"))
cat("Recall (YES):", Recall(actual2, pred2, positive = "YES"))
cat("F1 (NO):", F1_Score(actual2, pred2, positive = "NO"))
cat("F1 (YES):", F1_Score(actual2, pred2, positive = "YES"))





#Question 3

library(class)
library(MLmetrics)
library(ROCR)
library(e1071)


X_train = train[,c("age", "weight", "SOFA", "glu_min", "glu_max")]
Y_train = train[,c("death")]
X_test = test[,c("age", "weight", "SOFA", "glu_min", "glu_max")]
Y_test = test[, c("death")]

summary(X_train)
summary(X_test)


X_train_scaled = scale(X_train)
X_test_scaled  = scale(X_test)




k_values = c(1,2,3)

for (k in k_values) {
  pred <- knn(X_train_scaled, X_test_scaled, Y_train, k = k)
  
  
  cat("k =", k, "\n")
  print(ConfusionMatrix(pred, Y_test))
  cat("Accuracy:",  Accuracy(pred, Y_test), "\n")
  cat("Precision:", Precision(Y_test, pred, positive = "YES"), "\n")
  cat("Recall:",    Recall(Y_test, pred, positive = "YES"), "\n")
  cat("F1:",        F1_Score(Y_test, pred, positive = "YES"), "\n")
}





#SVM

train_svm<- data.frame(X_train_scaled, death = Y_train)
test_svm <- data.frame(X_test_scaled, death = Y_test)


svm_linear <- svm(death ~ ., kernel = "linear", type="C-classification", data = train_svm, probability = TRUE)
pred_linear <- predict(svm_linear, test_svm, probability = TRUE)

print(ConfusionMatrix(pred_linear, Y_test))
cat("Accuracy:",Accuracy(pred_linear, Y_test))
cat("Precision:",Precision(Y_test, pred_linear, positive = "YES"))
cat("Recall:",Recall(Y_test, pred_linear, positive = "YES"))
cat("F1:",F1_Score(Y_test, pred_linear, positive = "YES"))




svm_radial <- svm(death ~ ., kernel = "radial", type="C-classification", data = train_svm, probability = TRUE)
pred_radial <- predict(svm_radial, test_svm, probability = TRUE)

print(ConfusionMatrix(pred_radial, Y_test))
cat("Accuracy:",Accuracy(pred_radial, Y_test))
cat("Precision:",Precision(Y_test, pred_radial, positive = "YES"))
cat("Recall:",Recall(Y_test, pred_radial, positive = "YES"))
cat("F1:",F1_Score(Y_test, pred_radial, positive = "YES"))




svm_poly <- svm(death ~ ., kernel = "polynomial",type="C-classification",data = train_svm, probability = TRUE)
pred_poly <- predict(svm_poly, test_svm, probability = TRUE)

print(ConfusionMatrix(pred_poly, Y_test))
cat("Accuracy:",Accuracy(pred_poly, Y_test))
cat("Precision:",Precision(Y_test, pred_poly, positive = "YES"))
cat("Recall:",Recall(Y_test, pred_poly, positive = "YES"))
cat("F1:",F1_Score(Y_test, pred_poly, positive = "YES"))




prob_rbf <- attr(pred_radial, "probabilities")[, "YES"]
pred_obj_rbf <- prediction(prob_rbf, Y_test)
roc_rbf <- performance(pred_obj_rbf, "tpr", "fpr")

performance(pred_obj_rbf, "auc")
unlist(performance(pred_obj_rbf, "auc")@y.values[1])



pred_knn <- knn(X_train_scaled, X_test_scaled, Y_train, k = 2, prob = TRUE)
pred_obj_knn <- prediction(prob_knn, Y_test)
roc_knn <- performance(pred_obj_knn, "tpr", "fpr")
unlist(performance(pred_obj_knn, "auc")@y.values[1])



plot(roc_rbf, col = "blue", lwd = 2,
     main = "ROC Curves for kNN and SVM (RBF)")
plot(roc_knn, col = "red", lwd = 2, add = TRUE)

abline(0, 1, col = "grey")

legend("bottomright",
       legend = c("SVM RBF", "kNN (k=1)"),
       col = c("blue", "red"),
       lwd = 2)




#Question 4
library(cluster)

X <- train[, c("age", "rr_min", "rr_max", "spo2")]
summary(X)
X_scaled <- scale(X)


SSE <- c()
for (i in 1:10) {
  SSE[i] <- kmeans(X_scaled, centers = i)$tot.withinss
}

plot(1:10, SSE, type="b", xlab="Number of Clusters", ylab="SSE")




mean_silhouette <- c()

for (i in 2:10) {
  kmodel <- kmeans(X_scaled, centers = i)
  model_silhouette <- silhouette(kmodel$cluster, dist(X_scaled))
  mean_silhouette[i] <- mean(model_silhouette[, 3])
}

plot(2:10, mean_silhouette[2:10], type="b", xlab="Number of clusters", ylab="Average silhouette width")



finalmodel = kmeans(X_scaled, centers = 2)
finalmodel$centers
finalmodel$cluster


cohesion = finalmodel$tot.withinss
separation = finalmodel$betweenss

model_sil <- silhouette(finalmodel$cluster, dist(X_scaled))
plot(model_sil)
mean(model_sil[,3])


table(finalmodel$cluster, train$sepsis)
prop.table(table(finalmodel$cluster, train$sepsis), 1)

table(finalmodel$cluster, train$death)
prop.table(table(finalmodel$cluster, train$death), 1)




#Question 5
library(cluster)


dataq5 <- train[, c("weight", "SOFA", "wbc_min", "wbc_max", "hr")]
summary(dataq5)

dataq5$hr <- factor(dataq5$hr, levels = c("LOW", "NORMAL", "HIGH"))
dataq5$hr_num <- as.numeric(dataq5$hr)

X <- dataq5[, c("weight", "SOFA", "wbc_min", "wbc_max", "hr_num")]
X_scaled <- scale(X)
d <- dist(X_scaled)


hc_single   <- hclust(d, method = "single")
plot(hc_single)
hc_complete <- hclust(d, method = "complete")
plot(hc_complete)
hc_average  <- hclust(d, method = "average")
plot(hc_average)
hc_ward     <- hclust(d, method = "ward.D2")
plot(hc_ward)



slc = c()

for (k in 2:10){
  clusters = cutree(hc_ward, k = k)
  slc[k] = mean(silhouette(clusters, d)[, 3])
}

plot(2:10, slc[2:10], type="b",
     xlab="Number of Clusters",
     ylab="Silhouette")



clusters2 <- cutree(hc_ward, k = 2)
table(clusters2)


clusters3 <- cutree(hc_ward, k = 3)
plot(hc_ward)
rect.hclust(hc_ward, k = 3)
table(clusters3)

table(clusters3, dataq5$hr)
aggregate(cbind(weight, SOFA, wbc_min, wbc_max, hr_num) ~ clusters3, 
          data = dataq5, mean)



clusters4 <- cutree(hc_ward, k = 4)
table(clusters4)




#Question 6
library(dbscan)

dataq6 <- train[, c("glu_min", "glu_max", "wbc_min", "wbc_max", "SOFA")]
summary(dataq6)
data6_scaled <- scale(dataq6)


knndist = kNNdist(data6_scaled, k = 10)
plot(sort(knndist), type = 'l', xlab = "Points sorted by distance", ylab = "10-NN distance")



model = dbscan(data6_scaled, eps = 2, minPts = 10)
plot(data6_scaled, col = model$cluster + 1, pch = ifelse(model$cluster, 1, 4))

table(model$cluster)


outliers <- train[model$cluster == 0, ]
colMeans(outliers[, c("glu_min","glu_max","wbc_min","wbc_max","SOFA")])
colMeans(train[, c("glu_min","glu_max","wbc_min","wbc_max","SOFA")])

table(model$cluster == 0, train$mech_vent)
prop.table(table(model$cluster == 0, train$mech_vent), 1)


table(model$cluster == 0, train$death)
prop.table(table(model$cluster == 0, train$death), 1)
