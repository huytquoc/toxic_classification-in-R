# Input Data processing
library(readr)
library(caret)
setwd("~/Google Drive/STUDY/NYC Data Science/Projects/R_Toxic_Comments")
train <- read_csv("./input/train.csv")
submission <- read_csv("./input/test.csv")

source('./data_process.R')

# Data source
train_corpus <- build_corpus(train)
submission_corpus <- build_corpus(submission)

library(rJava)
library(RWeka)

# Word matrices of training and submission
NgramTokenizer1 <- function(x) {RWeka::NGramTokenizer(x, RWeka::Weka_control(min=1, max=1))}
dtm1 <- DocumentTermMatrix(train_corpus, control=list(tokenize=NgramTokenizer1, stemming = TRUE))
dtm1_999 <- removeSparseTerms(dtm1, 0.999)

dtm1_sub <- DocumentTermMatrix(submission_corpus, control=list(tokenize=NgramTokenizer1, stemming = TRUE))
dtm1_sub_999 <- removeSparseTerms(dtm1_sub, 0.999)

findFreqTerms(dtm1_999, 500)
freq_words = findFreqTerms(dtm1_999, 500)
str(freq_words)

# training Word matrix
dtm_matrix <- as.matrix(dtm1_999)
dtm_sub_matrix <- as.matrix(dtm1_sub_999)

# training Target class
labels <- colnames(train)[3:8]
target <- train[,labels]

dtm_train <- dtm_matrix[,freq_words]
dtm_sub <- dtm_sub_matrix[,freq_words]

predicted_sub <- matrix(0, nrow = nrow(dtm_sub), ncol = ncol(target))
colnames(predicted_sub) <- labels

# Cleaning no longer used data
rm(dtm_matrix,dtm1,dtm1_999,train_corpus, submission_corpus,
   dtm1_sub_999, dtm1_sub,dtm_sub_matrix,
   train, submission)

### 'toxic' class dataset
dataset_toxic <- toxic_class_dataset(dtm_train,target,'toxic')

### Split Train and Test datasets
set.seed(3456)
library(caret)
library(dplyr)
trainIndex <- createDataPartition(dataset_toxic$Class, p = .75, 
                                  list = FALSE, 
                                  times = 1)
train_toxic <- dataset_toxic[trainIndex,]
test_toxic <- dataset_toxic[-trainIndex,]


## Pre-processing for Keras training
desc <- train_toxic %>% 
  select(-Class) %>% 
  get_desc()

X_train <- train_toxic %>%
  select(-Class) %>%
  #normalization_minmax(desc) %>%
  as.matrix()
y_train <- train_toxic$Class

X_test <- test_toxic %>%
  select(-Class) %>%
  #normalization_minmax(desc) %>%
  as.matrix()
y_test <- test_toxic$Class

## Defining metrices
library(qdap)
library(keras)


metric_sensitivity <- function(y_true, y_pred){
  true_positives = k_sum(k_round(k_clip(y_true * y_pred, 0, 1)))
  possible_positives = k_sum(k_round(k_clip(y_true, 0, 1)))
  return(true_positives / (possible_positives + k_epsilon())) 
}

metric_specificity <- function(y_true, y_pred){
  true_negatives = k_sum(k_round(k_clip((1-y_true) * (1-y_pred), 0, 1)))
  possible_negatives = k_sum(k_round(k_clip(1-y_true, 0, 1)))
  return(true_negatives / (possible_negatives + k_epsilon())) 
}

metric_precision <- function(y_true, y_pred){
  # #Precision metric.
  # 
  # Only computes a batch-wise average of precision.
  # 
  # Computes the precision, a metric for multi-label classification of
  # how many selected items are relevant.
  
  true_positives = k_sum(k_round(k_clip(y_true * y_pred, 0, 1)))
  predicted_positives = k_sum(k_round(k_clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + k_epsilon())
  return(precision)
}

metric_recall <- function(y_true, y_pred){
  # """Recall metric.
  # 
  # Only computes a batch-wise average of recall.
  # 
  # Computes the recall, a metric for multi-label classification of
  # how many relevant items are selected.
  # """
  
  true_positives = k_sum(k_round(k_clip(y_true * y_pred, 0, 1)))
  possible_positives = k_sum(k_round(k_clip(y_true, 0, 1)))
  recall = true_positives / (possible_positives + k_epsilon())
  return(recall)
}

metric_f1 <- function(y_true, y_pred){
  precision = metric_precision(y_true, y_pred)
  recall = metric_recall(y_true, y_pred)
  return(2*((precision*recall)/(precision+recall)))
}

  



t = 0.5 #both classes are equally important
n_non_toxic = sum(y_train==0)
n_yes_toxic = sum(y_train==1)
n_total = length(y_train)

X_train_norm <- normalize(X_train)
X_test_norm <- normalize(X_test)

#####
model <- keras_model_sequential() %>%
  layer_dense(units = 100, activation = "relu", input_shape = c(ncol(X_train))) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 50, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 25, activation = "relu") %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 1, activation = "sigmoid")

summary(model)

model %>% compile(
  optimizer = "rmsprop", #"adam", #
  loss = "binary_crossentropy",
  metrics = c("accuracy",
              "precision" = metric_precision,
              "recall" = metric_recall)
              #"sensitivity" = metric_sensitivity,
              #"specificity" = metric_specificity)#,
  #"toxic_class_performing" = metric_single_class_accuracy)
)

checkpoint <- callback_model_checkpoint(
  filepath = "modelv3.hdf5", 
  save_best_only = TRUE, 
  period = 1,
  verbose = 1
)

early_stopping <- callback_early_stopping(patience = 5)

### Fitting Keras model
history <- model %>% fit(
  X_train_norm,
  y_train,
  epochs = 100,
  batch_size = 64,
  #callbacks = callback_tensorboard("logs/run"),
  #validation_data = list(X_test_tx_val, y_test_tx_val),
  validation_split = 0.2,
  class_weight = list("0" = 1,"1" = n_non_toxic/n_yes_toxic),
  callbacks = list(checkpoint, early_stopping, callback_tensorboard("logs/run"))
)

#model_saved <- load_model_hdf5("modelv3.hdf5")#, custom_objects = NULL, compile = TRUE)

summary(model)
plot(history)

library(ROCR)
library(Metrics)
#training performance
y_train_pred <- model %>% predict_classes(X_train_norm)
y_train_pred_prob <- model %>% predict_proba(X_train_norm)
result_train <- confusionMatrix(y_train_pred, y_train)
result_train
auc(y_train, y_train_pred_prob)
#result_train$byClass[7] #F1

#testing performance
y_test_pred <- model %>% predict_classes(X_test_norm)
y_test_pred_prob <- model %>% predict_proba(X_test_norm)
result_test <- confusionMatrix(y_test_pred, y_test)
result_test
auc(y_test, y_test_pred_prob)
library(pROC)
plot(roc(y_test, y_test_pred_prob))


### output consideration
possible_k <- seq(0, 0.5, length.out = 100)
precision <- sapply(possible_k, function(k) {
  predicted_class <- as.numeric(y_test_pred_prob > k)
  sum(predicted_class == 1 & y_test == 1)/sum(predicted_class)
})

qplot(possible_k, precision, geom = "line") + labs(x = "Threshold", y = "Precision")

recall <- sapply(possible_k, function(k) {
  predicted_class <- as.numeric(y_test_pred_prob > k)
  sum(predicted_class == 1 & y_test == 1)/sum(y_test)
})
qplot(possible_k, recall, geom = "line") + labs(x = "Threshold", y = "Recall")

#####



### Resampling the Training dataset
train_toxic_down <- downSample(x = train_toxic[, -ncol(train_toxic)],
                         y = as.factor(train_toxic$Class))
table(train_toxic_down$Class) 

### Feature selections
ga_ctrl <- gafsControl(functions = rfGA,
                       method = "repeatedcv",
                       repeats = 5)

## Use the same random number seed as the RFE process
## so that the same CV folds are used for the external
## resampling.
library(doMC)
registerDoMC(cores = 5)

set.seed(111)
rf_ga <- gafs(x = train_toxic_down[,-ncol(train_toxic_down)], y = train_toxic_down$Class,
              iters = 20,
              gafsControl = ga_ctrl)
rf_ga

plot(rf_ga) + theme_bw()

### Training Process
fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10,
  savePredictions = TRUE)

set.seed(111)

glmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                        n.trees = (1:30)*50, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

glm_fit <- train(Class ~ .,  data=train_toxic_down, method="glm", family="binomial",
                 trControl = fitControl, tuneLength = 5)

pred = predict(glm_fit, newdata=train_toxic_down[,-ncol(train_toxic_down)])
confusionMatrix(data=pred, train_toxic_down$Class)

###

