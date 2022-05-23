# Supervised model
# Text analysis on scientific abstracts

# 0. ETL
library(sampler)
library(readxl)
library(tm)
library(tidytext)
library(quanteda)
library(quanteda.textstats)
library(quanteda.textplots)
library(quanteda.textmodels)
library(lubridate)
library(caret)
library(ROCR)

set.seed(41)

#### HELPER FUNCTIONS for calculating precision and recall
error_metric=function(CM)
{
  TN = CM[1,1]
  TP = CM[2,2]
  FP = CM[1,2]
  FN = CM[2,1]
  prec  <- (TP)/(TP+FP)
  accu <- (TP+TN)/(TP+TN+FP+FN)
  recall <- (TP)/(TP+FN)
  print(paste("Precision of model: ",round(prec,3)))
  print(paste("Accuracy of model: ",round(accu,3)))
  print(paste("Recall of model: ",round(recall,3)))
}
####

# load data frame 
df_1 <- read_excel("H:\\Meta-Rep\\R Code\\Lit Classifier\\Data\\CODE.xlsx")
df_1 <- data.frame(doc_id=row.names(df_1),
                 text=df_1$Abstract, relevant = df_1$included)

df_2 <- read_excel("H:\\Meta-Rep\\R Code\\Lit Classifier\\Data\\Test_Data.xlsx")
df_2 <- data.frame(doc_id=row.names(df_2),
                      text=df_2$Abstract, relevant = df_2$included)

df_3 <- read_excel("H:\\Meta-Rep\\R Code\\Lit Classifier\\Data\\Results_Sampled_Dataset_3.xlsx")
df_3 <- data.frame(doc_id=row.names(df_3),
                      text=df_3$Abstract, relevant = df_3$included)

df_4 <- read_excel("H:\\Meta-Rep\\R Code\\Lit Classifier\\Data\\Phil_Review_Data.xlsx")
df_4 <- data.frame(doc_id=row.names(df_4),
                   text=df_4$Abstract, relevant = df_4$Relevant)
df_full <- rbind(df_1,df_2,df_3, df_4)

df_full$doc_id <- NULL

# delete duplicate rows (40 duplicates in this case)
df_full <- df_full[!duplicated(df_full$text), ]

# DFM cleaning and stopword removal
stopwords_manual <- c("social", "media", "news", "study", "article", "research", "results", "can", "findings", "new", "approach", 
                      "two", "also", "using", "may", "used", "although", "one", "can", "use", "well", "three", "paper",
                      "e.g", "i.e", "need", "however", "first", "whether", "often", "likely", "second", "yet") #based on doc_frequency

corp_review <- corpus(df_full, text_field = "text")

toks_reviews <- tokens(corp_review, remove_punct = TRUE)
dfmat_reviews <- dfm(toks_reviews, remove_punct = TRUE, remove_numbers = TRUE, 
                     remove_symbols = TRUE) %>%
  dfm_remove(stopwords("english")) %>%
  dfm_remove(stopwords_manual) %>%
  dfm_trim(min_docfreq = 3)

# test train split
smp_size <- floor(0.75 * nrow(dfmat_reviews))

train_ind <- sample(seq_len(nrow(dfmat_reviews)), size = smp_size)

dfm_train <- dfmat_reviews[train_ind, ]
dfm_test <- dfmat_reviews[-train_ind, ]

# classifier (Naive Bayes, SVM, linear SVM)
my_nb_classifier <-  textmodel_nb(dfm_train, 
                                  docvars(dfm_train, "relevant"))

my_svm_classifier <- textmodel_svm(dfm_train, probability = TRUE, 
                                   docvars(dfm_train, "relevant"))

my_svmlin_classifier <- textmodel_svmlin(dfm_train, 
                                 docvars(dfm_train, "relevant"))

# prediction
predicted_nb  <- predict(my_nb_classifier,newdata=dfm_test)
predicted_svm  <- predict(my_svm_classifier,newdata=dfm_test)
predicted_svmlin  <- predict(my_svmlin_classifier,newdata=dfm_test)

# performance measures
actual <- docvars(dfm_test, "relevant")

ctab_nb <- table(predicted_nb, actual)
confusionMatrix(ctab_nb, positive = "1")
error_metric(ctab_nb)

ctab_svm <- table(predicted_svm, actual)
confusionMatrix(ctab_svm, positive = "1")
error_metric(ctab_svm)

ctab_svm_lin <- table(predicted_svmlin, actual)
confusionMatrix(ctab_svm_lin, positive = "1")
error_metric(ctab_svm_lin)

# ensemble classifier
df_ens <- cbind.data.frame(pred_nb = as.numeric(predicted_nb), pred_svm = as.numeric(predicted_svm), pred_lin_svm = as.numeric(predicted_svmlin))
df_ens$pred_ens <- floor(((df_ens$pred_nb-1) + df_ens$pred_svm + df_ens$pred_lin_svm)/2)

ctab_ens <- table(df_ens$pred_ens, actual)
confusionMatrix(ctab_ens, positive = "1")
error_metric(ctab_ens)


# ROC and AUC plots
# Naive Bayes
nb_quanteda.prob <- predict(my_nb_classifier, type= "probability",  newdata = dfm_test)
pred_nb_quanteda <- prediction(as.numeric(nb_quanteda.prob[,2]), actual)
perf_nb_quanteda <- performance(pred_nb_quanteda,"tpr","fpr")  
plot( perf_nb_quanteda, col="darkblue", main="ROC and AUC for Naive Bayes")

abline(coef = c(0.0,1.0))
perf_AUC <- performance(pred_nb_quanteda, "auc")
AUC <- perf_AUC@y.values[[1]]
text(0.8,0.2,paste("AUC = ",format(AUC, digits=5, scientific=FALSE)))
jpeg(file="ROC and AUC NB.jpeg")

# SVM
svm_quanteda.prob <- predict(my_svm_classifier, type = "class",  newdata = dfm_test)
pred_svm_quanteda <- prediction(as.numeric(svm_quanteda.prob), actual)
perf_svm_quanteda <- performance(pred_svm_quanteda,"tpr","fpr")  
plot(perf_svm_quanteda, col="darkblue", main="ROC for SVM")

abline(coef = c(0.0,1.0))
perf_AUC <- performance(pred_svm_quanteda, "auc")
AUC <- perf_AUC@y.values[[1]]
text(0.8,0.2,paste("AUC = ",format(AUC, digits=5, scientific=FALSE)))
jpeg(file="ROC and AUC SVM.jpeg")

# SVM linear
svm_lin_quanteda.prob <- predict(my_svmlin_classifier, type = "class",  newdata = dfm_test)
pred_svm_lin_quanteda <- prediction(as.numeric(svm_lin_quanteda.prob), actual)
perf_svm_lin_quanteda <- performance(pred_svm_lin_quanteda,"tpr","fpr")  
plot(perf_svm_lin_quanteda, col="darkblue", main="ROC for linear SVM")

abline(coef = c(0.0,1.0))
perf_AUC <- performance(pred_svm_lin_quanteda, "auc")
AUC <- perf_AUC@y.values[[1]]
text(0.8,0.2,paste("AUC = ",format(AUC, digits=5, scientific=FALSE)))
jpeg(file="ROC and AUC SVM linear.jpeg")

####### 
#######

# percentage of not-relevant vs relevant
# Goal:
table(df_full$relevant)[1]/nrow(df_full)
# Where we're at:
table(docvars(dfm_train)$relevant)/(table(docvars(dfm_train)$relevant)[1] + table(docvars(dfm_train)$relevant)[2])

# ratio in the test sample:
table(docvars(dfm_test)$relevant)/(table(docvars(dfm_test)$relevant)[1] + table(docvars(dfm_test)$relevant)[2])

# Perform preprocessing for train and test data sets
# remove all stopwords
corp_review_train <- corpus(train_sample, text_field = "text")

toks_reviews_train <- tokens(corp_review_train, remove_punct = TRUE)
dfmat_reviews_train <- dfm(toks_reviews_train, remove_punct = TRUE, remove_numbers = TRUE, 
                     remove_symbols = TRUE) %>%
  dfm_remove(stopwords("english")) %>%
  dfm_remove(stopwords_manual) %>%
  dfm_trim(min_docfreq = 3)

corp_review_test <- corpus(test_sample, text_field = "text")

toks_reviews_test <- tokens(corp_review_test, remove_punct = TRUE)
dfmat_reviews_test <- dfm(toks_reviews_test, remove_punct = TRUE, remove_numbers = TRUE, 
                           remove_symbols = TRUE) %>%
  dfm_remove(stopwords("english")) %>%
  dfm_remove(stopwords_manual) %>%
  dfm_trim(min_docfreq = 3)

#####
# End of the code