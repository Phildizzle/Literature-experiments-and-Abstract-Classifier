# Text analysis on scientific abstracts

# 0. ETL
library(readxl)
library(tm)
library(tidytext)
library(quanteda)
library(quanteda.textstats)
library(quanteda.textplots)
library(quanteda.corpora)
library(lubridate)
library(caret)

# read data frame and create simpler version
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

# accuracy/precision helper function
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

#######
# for loop for model transparency, varying parameter is the sent cutoff value
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

# 3. Text statistics and other shenanigans
# 3.1 clear stopwords
# identify words with max 3 mentions
dfmat_reviews <- dfm(toks_reviews)
tstat_freq <- textstat_frequency(-dfmat_reviews, n =3200)
stopwords_prop <-  list(tstat_freq$feature)

# identify manual stopwords
stopwords_manual <- c("social", "media", "news", "study", "article", "research", "results", "can", "findings", "new", "approach", 
                      "two", "also", "using", "may", "used", "although", "one" )

# remove all stopwords
toks_reviews <- tokens(corp_review, remove_punct = TRUE)
dfmat_reviews <- dfm(toks_reviews) %>%
  dfm_remove(stopwords_prop) %>%
  dfm_remove(stopwords("english")) %>%
  dfm_remove(stopwords_manual)

#textstat experiments
# plot 30 most frequent words
dfmat_reviews %>% 
  textstat_frequency(n = 30) %>% 
  ggplot(aes(x = reorder(feature, frequency), y = frequency)) +
  geom_point() +
  coord_flip() +
  labs(x = NULL, y = "Frequency") +
  theme_minimal()

# obligatory worcloud
textplot_wordcloud(dfmat_reviews, max_words = 100)


tstat_key <- textstat_keyness(dfmat_reviews)
textplot_keyness(tstat_key)


# 4. Supervised ML model
dfm_train <- corp_review %>% 
  tokens() %>%
  dfm() %>% 
  dfm_trim(min_docfreq=0.01, docfreq_type="prop")

myclassifier <-  textmodel_nb(dfm_train, 
                            docvars(dfm_train, "relevant"))

# in sample to test the model
predicted  <- predict(myclassifier,newdata=dfm_train)
actual <- docvars(dfm_train, "relevant")

results <- list()
for (label in c("pos", "neg")) {
  results[[label]] = tibble(
    Precision=Precision(actual, predicted, label),
    Recall=Recall(actual, predicted, label),
    F1=F1_Score(actual, predicted, label))
}
bind_rows(results, .id="label")  

CM <- table(predicted, actual)

error_metric(CM)

# out of sample

df_test <- read_excel("H:\\Meta-Rep\\R Code\\Lit Classifier\\Data\\CODE.xlsx")
df_test <- data.frame(doc_id=row.names(df),
                 text=df$Abstract, relevant = df$included)





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

###################################################
###################################################
# Old code

# make dfm, remove stopwords, create raw counts of pos words  

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
