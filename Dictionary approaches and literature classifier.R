# Text analysis on scientific abstracts
# 0. ETL
library(readxl)
library(tm)
library(tidytext)
library(quanteda)
library(quanteda.textstats)
library(quanteda.textplots)
library(quanteda.textmodels)
library(lubridate)
library(caret)

#### HELPER FUNCTIONS
error_metric=function(CM)
{
  TN =CM[1,1]
  TP =CM[2,2]
  FP =CM[1,2]
  FN =CM[2,1]
  precision  <- (TP)/(TP+FP)
  accuracy_model <- (TP+TN)/(TP+TN+FP+FN)
  recall <- (TP)/(TP+FN)
  print(paste("Precision value of the model: ",round(precision,3)))
  print(paste("Accuracy of the model: ",round(accuracy_model,3)))
  print(paste("Recall of the model: ",round(recall,3)))
}
####

# load data frame 
df <- read_excel("H:\\Meta-Rep\\R Code\\Lit Classifier\\Data\\CODE.xlsx")
df <- data.frame(doc_id=row.names(df),
                 text=df$Abstract, relevant = df$included)

corp_review <- corpus(df, text_field = "text")

# 1. Dictionary exercise
# sentiment dictionary
poswords <- "H:\\Meta-Rep\\R Code\\Lit Classifier\\Data\\Positive.txt"
negwords <- "H:\\Meta-Rep\\R Code\\Lit Classifier\\Data\\Negative.txt"
pos <- scan(poswords, what="list")
neg <- scan(negwords, what="list")
sentimentdict <- dictionary(list(pos=pos, neg=neg))

# normal dictionary
scores <- corp_review  %>% 
  tokens() %>%
  dfm() %>% 
  dfm_lookup(sentimentdict) %>% 
  convert(to="data.frame")  %>% 
  mutate(sent = pos - neg)
head(scores)

##### accuracy/precision helper function
error_metric=function(CM)
{
  TN =CM[1,1]
  TP =CM[2,2]
  FP =CM[1,2]
  FN =CM[2,1]
  precision  <- (TP)/(TP+FP)
  accuracy_model <- (TP+TN)/(TP+TN+FP+FN)
  recall <- (TP)/(TP+FN)
  print(paste("Precision value of the model: ",round(precision,3)))
  print(paste("Accuracy of the model: ",round(accuracy_model,3)))
  print(paste("Recall of the model: ",round(recall,3)))
}

#####

# for loop for model transparency, varying parameter is the sentiment cutoff value
values <- seq(from = -5, to = 5)
# -5 to 5 since around 70-80% of the distribution falls in between see hist(scores$sent)

scores$relevant <- ifelse(scores$sent >0, 1, 0)
CM <- table(df$relevant, scores$relevant)

for (i in values) {
  scores$relevant <- ifelse(scores$sent >i-1, 1, 0)
  CM <- table(df$relevant, scores$relevant)
  print(paste("Model with sent cutoff at: ", i))
  error_metric(CM)
  print(paste("________________"))
}

# 2. Weighted dictionary exercise

# weighted dictionary (weighted DFM with only positive weights)
# including negative weights changes almost nothing
poswords_list <- c("agent-based", "analytics", "automated", "computational", "corpus", "data", "data-driven", 
                   "digital", "digitalized", "language", "learning", "machine", "method", "methods", "methodology", "mining", "model",
                   "model-based", "models", "modeling", "network", "quantitative", "semantic", "semi-automated", "sentiment", 
                   "supervised", "Twitter", "topic", "unsupervised")

# subjective weightinga
pos_weights <- c(1, 1, 1, 5, 5, 1, 3, 1, 1, 1, 1, 3, 1, 1, 1, 3, 1, 3, 1, 3, 5, 5, 3, 5, 5, 5, 1, 3, 5)
names(pos_weights) <- poswords_list

# reload DFM since we weigh the DFM and not the dictionary itself
toks_reviews <- tokens(corp_review, remove_punct = TRUE)
dfmat_reviews <- dfm(toks_reviews) %>%
  dfm_remove(stopwords("english")) %>%
  dfm_select(pattern = names(pos_weights))

dfm_reviews_weighted <- dfm_weight(dfmat_reviews, weight = pos_weights)
scores_weighted <- dfm_lookup(dfm_reviews_weighted, sentimentdict)


scores <- dfm_reviews_weighted  %>% 
  dfm_lookup(sentimentdict) %>% 
  convert(to="data.frame")  %>% 
  mutate(sent = pos - neg)
head(scores)

values <- seq(from = 5, to = 16)
# values from 5-16 

for (i in values) {
  scores$relevant <- ifelse(scores$sent >i-1, 1, 0)
  CM <- table(df$relevant, scores$relevant)
  print(paste("Model with sent cutoff at: ", i))
  error_metric(CM)
  print(paste("________________"))
}

# weighting definitely improves accuracy measures

# 3. Text statistics and other shenanigans
# 3.1 clear stopwords
# identify words with max 3 mentions
toks_reviews <- tokens(corp_review, remove_punct = TRUE)
dfmat_reviews <- dfm(toks_reviews)
tstat_freq <- textstat_frequency(dfmat_reviews)
stopwords_prop <-  list(tstat_freq$feature)

# identify manual stopwords
stopwords_manual <- c("social", "media", "news", "study", "article", "research", "results", "can", "findings", "new", "approach", 
                      "two", "also", "using", "may", "used", "although", "one", "can", "use", "well", "three", "paper") #based on doc_frequency

# remove all stopwords
toks_reviews <- tokens(corp_review, remove_punct = TRUE)
dfmat_reviews <- dfm(toks_reviews, remove_punct = TRUE, remove_numbers = TRUE, 
                     remove_symbols = TRUE) %>%
  #  dfm_remove(stopwords_prop) %>%
  dfm_remove(stopwords("english")) %>%
  dfm_remove(stopwords_manual) %>%
  dfm_trim(min_docfreq = 3)

tstat_freq <- textstat_frequency(dfmat_reviews)

# Keyness
dfmat_reviews_grouped <- dfm_group(dfmat_reviews, groups = relevant)
dfmat_reviews_grouped
review_keyness <- textstat_keyness(dfmat_reviews_grouped)
textplot_keyness(review_keyness, n = 25)

textstat_keyness(dfmat_reviews_grouped, target = "0") %>%
  textplot_keyness(n = 25, color = c("green", "pink"))

textstat_keyness(dfmat_reviews_grouped, target = "1") %>%
  textplot_keyness(n = 25, color = c("green", "pink"))

#textstat experiments
# plot 30 most frequent words
#dfmat_reviews %>% 
#  textstat_frequency(n = 30) %>% 
#  ggplot(aes(x = reorder(feature, frequency), y = frequency)) +
#  geom_point() +
#  coord_flip() +
#  labs(x = NULL, y = "Frequency") +
#  theme_minimal()

# obligatory worcloud
textplot_wordcloud(dfmat_reviews, max_words = 100)

dfmat_reviews_grouped <- dfm_group(dfmat_reviews, groups = relevant)
dfmat_reviews_grouped


# 4. Supervised ML model
toks_reviews <- tokens(corp_review, remove_punct = TRUE)

stopwords_manual <- c("social", "media", "news", "study", "article", "research", "results", "can", "findings", "new", "approach", 
                      "two", "also", "using", "may", "used", "although", "one", "can", "use", "well", "three", "paper") #based on doc_frequency

dfm_train <- dfm(toks_reviews, remove_punct = TRUE, remove_numbers = TRUE, 
                 remove_symbols = TRUE) %>%
  #  dfm_remove(stopwords_prop) %>%
  dfm_remove(stopwords("english")) %>%
  dfm_remove(stopwords_manual) %>%
  dfm_trim(min_docfreq=0.1, docfreq_type="prop")


my_nb_classifier <-  textmodel_nb(dfm_train, 
                                  docvars(dfm_train, "relevant"))

my_svm_classifier <- textmodel_svm(dfm_train, 
                                   docvars(dfm_train, "relevant"))

my_lr_classifier <- textmodel_lr(dfm_train, 
                                 docvars(dfm_train, "relevant"))
# in sample to test the model

# Naive Bayes performance
predicted_nb  <- predict(my_nb_classifier,newdata=dfm_train)
actual <- docvars(dfm_train, "relevant")

results <- list()
for (label in c("pos", "neg")) {
  results[[label]] = tibble(
    Precision=Precision(actual, predicted_nb, label),
    Recall=Recall(actual, predicted_nb, label),
    F1=F1_Score(actual, predicted_nb, label))
}

CM_nb <- table(predicted_nb, actual)

error_metric(CM_nb)

# SVM performance
predicted_svm  <- predict(my_svm_classifier,newdata=dfm_train)
actual <- docvars(dfm_train, "relevant")

results <- list()
for (label in c("pos", "neg")) {
  results[[label]] = tibble(
    Precision=Precision(actual, predicted_svm, label),
    Recall=Recall(actual, predicted_svm, label),
    F1=F1_Score(actual, predicted_svm, label))
}

CM_svm <- table(predicted_svm, actual)

error_metric(CM_svm)

# Logistic Regression performance
predicted_lr  <- predict(my_lr_classifier,newdata=dfm_train)
actual <- docvars(dfm_train, "relevant")

results <- list()
for (label in c("pos", "neg")) {
  results[[label]] = tibble(
    Precision=Precision(actual, predicted_lr, label),
    Recall=Recall(actual, predicted_lr, label),
    F1=F1_Score(actual, predicted_lr, label))
}

CM_lr <- table(predicted_lr, actual)

error_metric(CM_lr)


# out of sample performance
# Please keep in mind that the classification rates between train and test data vary considerably!
# Train: Yes = 73 vs No = 77
# Test: Yes = 131 vs No 49

df_test <- read_excel("H:\\Meta-Rep\\R Code\\Lit Classifier\\Data\\Test_Data.xlsx")
df_test <- data.frame(doc_id=row.names(df_test),
                      text=df_test$Abstract, relevant = df_test$included)

corp_review_test <- corpus(df_test, text_field = "text")

toks_reviews <- tokens(corp_review, remove_punct = TRUE)
dfm_train <- dfm(toks_reviews, remove_punct = TRUE, remove_numbers = TRUE, 
                 remove_symbols = TRUE) %>%
  #  dfm_remove(stopwords_prop) %>%
  dfm_remove(stopwords("english")) %>%
  dfm_remove(stopwords_manual) %>%
  dfm_trim(min_docfreq=0.1, docfreq_type="prop")

corp_review_test <- corpus(df_test, text_field = "text")

toks_reviews_test <- tokens(corp_review_test, remove_punct = TRUE)

dfm_test <- dfm(toks_reviews_test, remove_punct = TRUE, remove_numbers = TRUE, 
                remove_symbols = TRUE) %>%
  #  dfm_remove(stopwords_prop) %>%
  dfm_remove(stopwords("english")) %>%
  dfm_remove(stopwords_manual) %>%
  dfm_trim(min_docfreq=0.1, docfreq_type="prop")


predicted_nb_test  <- predict(my_nb_classifier,newdata=dfm_train)

predicted_svm_test  <- predict(my_svm_classifier,newdata=dfm_test)

predicted_lr_test  <- predict(my_lr_classifier,newdata=dfm_test)

# Naive Bayes performance
predicted_nb_test  <- predict(my_nb_classifier,newdata=dfm_test)
actual <- docvars(dfm_test, "relevant")

results <- list()
for (label in c("pos", "neg")) {
  results[[label]] = tibble(
    Precision=Precision(actual, predicted_nb_test, label),
    Recall=Recall(actual, predicted_nb_test, label),
    F1=F1_Score(actual, predicted_nb_test, label))
}

CM_nb_test <- table(predicted_nb_test, actual)

error_metric(CM_nb_test)

# SVM performance
predicted_svm  <- predict(my_svm_classifier,newdata=dfm_train)
actual <- docvars(dfm_train, "relevant")

results <- list()
for (label in c("pos", "neg")) {
  results[[label]] = tibble(
    Precision=Precision(actual, predicted_svm, label),
    Recall=Recall(actual, predicted_svm, label),
    F1=F1_Score(actual, predicted_svm, label))
}

CM_svm <- table(predicted_svm, actual)

error_metric(CM_svm)

# Logistic Regression performance
predicted_lr  <- predict(my_lr_classifier,newdata=dfm_train)
actual <- docvars(dfm_train, "relevant")

results <- list()
for (label in c("pos", "neg")) {
  results[[label]] = tibble(
    Precision=Precision(actual, predicted_lr, label),
    Recall=Recall(actual, predicted_lr, label),
    F1=F1_Score(actual, predicted_lr, label))
}

CM_lr <- table(predicted_lr, actual)

error_metric(CM_lr)



###################################################
###################################################
# Old code

# make dfm, remove stopwords, create raw counts of pos words  

#plot dtm frequency
dfmat_reviews %>% 
  textstat_frequency(n = 30) %>% 
  ggplot(aes(x = reorder(feature, frequency), y = frequency)) +
  geom_point() +
  coord_flip() +
  labs(x = NULL, y = "Frequency") +
  theme_minimal()

toks_reviews <- tokens(corp_review, remove_punct = TRUE) 
dfmat_reviews <- dfm(toks_reviews) %>% 
  dfm_remove(stopwords("english"))


tstat_dist <- as.dist(textstat_dist(dfmat_reviews))
clust <- hclust(tstat_dist)
plot(clust, xlab = "Distance", ylab = NULL)

toks_reviews <- tokens(corp_review, remove_punct = TRUE)
dfmat_reviews <- dfm(toks_reviews) %>%
  dfm_remove(stopwords("english")) %>%
  dfm_select(pattern = names(pos_weights))


scores <- corpus_sample(dfm_reviews_weighted, 150)  %>%
  dfm_lookup(sentimentdict) %>% 
  convert(to="data.frame")  %>% 
  mutate(sent = pos - neg)
head(scores)


toks_reviews <- tokens(corp_review, remove_punct = TRUE) 
dfmat_reviews <- dfm(toks_reviews) %>%
  dfm_remove(stopwords("english")) %>%
  dfm_select(pattern = names(pos_weights))

dfm_reviews_weighted <- dfm_weight(dfmat_reviews, weight = pos_weights)

sentimentdict <- dictionary(list(pos=pos, neg=neg))


scores <- dfm_lookup(dfm_reviews_weighted, sentimentdict)

scores <- corpus_sample(dfm_reviews_weighted, 150)  %>%
  dfm_lookup(sentimentdict) %>% 
  convert(to="data.frame")  %>% 
  mutate(sent = pos - neg)
head(scores)



###### confusion matrix



scores$relevant <- ifelse(scores$sent >15, 1, 0)
CM <- table(df$relevant, scores$relevant)


metrics <- error_metric(CM)

#######
# for loop version
values <- seq(from = 5, to = 16)


for (i in values) {
  scores$relevant <- ifelse(scores$sent >i-1, 1, 0)
  CM <- table(df$relevant, scores$relevant)
  print(paste("Model with sent cutoff at: ", i))
  error_metric(CM)
  print(paste("________________"))
}


#######




df_test <- read_excel("H:\\Meta-Rep\\R Code\\Lit Classifier\\Data\\CODE.xlsx")
