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