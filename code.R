library(tm)
library(textclean)
library(ggplot2)
library(ggwordcloud)
library(tensorflow)
library(keras)
library(kableExtra)
library(utf8)
library(spacyr)

df <- read.csv("./tweets_labelled_09042020_16072020.csv", sep = ";", encoding = "UTF-8")
df <- df[3:4]
df$sentiment[df$sentiment == "positive"] <- as.double(2)
df$sentiment[df$sentiment == "neutral"] <- as.double(1)
df$sentiment[df$sentiment == "negative"] <- as.double(0)

df$text[!utf8_valid(df$text)]
df_NFC <- utf8_normalize(df$text)
sum(df$text != df_NFC) # should be 0

# transformations
cleanUp <- function(CORPUS){
  CORPUS <- tm_map(CORPUS, content_transformer(tolower))
  CORPUS <- tm_map(CORPUS, content_transformer(function(x){gsub("[^\x01-\x7F]", "", x)}))
  CORPUS <- tm_map(CORPUS, content_transformer(function(x){replace_contraction(x)}))
  CORPUS <- tm_map(CORPUS, content_transformer(function(x){gsub("\\S*http+\\S*", "", x)}))
  CORPUS <- tm_map(CORPUS, content_transformer(function(x){gsub("&amp", "&", x)}))
  CORPUS <- tm_map(CORPUS, content_transformer(function(x){gsub("\\$\\d+", "", x)}))
  CORPUS <- tm_map(CORPUS, content_transformer(function(x){gsub("[.,;:+-=*?!/%{}()<>_~^']", "", x)}))
  CORPUS <- tm_map(CORPUS, content_transformer(removeNumbers))
  CORPUS <- tm_map(CORPUS, removeWords, stopwords("english"))
  CORPUS <- tm_map(CORPUS, stemDocument, language = "english")
  CORPUS <- tm_map(CORPUS, stripWhitespace)
  CORPUS <- tm_map(CORPUS, trimws)
}

df_clean <- cleanUp(Corpus(VectorSource(df$text)))
df$text[1] # before
df_clean[[1]]$content # after

# Histogram of length of tweets
tweets <- df_clean[[1]]$content
for (i in 2:length(df_clean)) {
  tweets <- append(tweets, df_clean[[i]]$content)
}

hist(nchar(tweets),
     main = "Histogram of tweet size",
     xlab = "Tweet size (number of characters)",
     ylab = "Ocurrences")

# Frequency of words, hastags and stocks
tdmTR <- TermDocumentMatrix(df_clean)
freq <- findFreqTerms(tdmTR, lowfreq = 50, highfreq = Inf)
freq <- as.matrix(tdmTR[freq,])
freq <- as.data.frame(freq)
freq <- as.data.frame(rowSums(freq))
colnames(freq) <- "num"
freq$word <- rownames(freq)
freq <- freq[order(freq$num, decreasing = T),]

freq_hastags <- freq[grepl('^#', freq$word),]
freq_stocks <- freq[grepl('^\\$', freq$word),]
freq_normal <- freq[(!grepl('^#', freq$word) & !grepl('^\\$', freq$word)),]

barplot(freq_hastags$num[1:10], names.arg = freq_hastags$word[1:10])
barplot(freq_stocks$num[1:10], names.arg = freq_stocks$word[1:10])
barplot(freq_normal$num[1:10], names.arg = freq_normal$word[1:10])

# WordCloud
options(repr.plot.width = 18, repr.plot.height = 8)
ggplot(freq_hastags, aes(label = word, size = num, color = num)) +
  geom_text_wordcloud(shape = "circle", rm_outside = TRUE, area_corr = F, rstep = .01, 
                      max_grid_size = 256, grid_size = 7, grid_margin = .4) +
  scale_size_area(max_size = 12.5) + theme_minimal() +
  scale_color_gradient(low = "darkgrey", high = "#53d1b1")

ggplot(freq_stocks, aes(label = word, size = num, color = num)) +
  geom_text_wordcloud(shape = "circle", rm_outside = TRUE, area_corr = F, rstep = .01, 
                      max_grid_size = 256, grid_size = 7, grid_margin = .4) +
  scale_size_area(max_size = 12.5) + theme_minimal() +
  scale_color_gradient(low = "darkgrey", high = "#53d1b1")

ggplot(freq_normal, aes(label = word, size = num, color = num)) +
  geom_text_wordcloud(shape = "circle", rm_outside = TRUE, area_corr = F, rstep = .01, 
                      max_grid_size = 256, grid_size = 7, grid_margin = .4) +
  scale_size_area(max_size = 12.5) + theme_minimal() +
  scale_color_gradient(low = "darkgrey", high = "#53d1b1")

#spacy_install()
spacy_initialize()

res <- lapply(tweets[1:100], spacy_parse, dependency = TRUE, nounphrase = TRUE)
df_tag <- res[[1]]
for (i in 2:length(res)){
  df_tag <- rbind(df_tag, res[[i]])
}

kable_styling(kable(df_tag[98:120, c(3:ncol(df_tag))]), font_size = 12)
spacy_finalize()

# LSTM
df_classification <- as.data.frame(df_clean$content)
colnames(df_classification) <- 'text'
df_classification$sentiment <- df$sentiment
df_classification <- df_classification[1:1300,]

set.seed(1)
split <- sample(c(T, F), nrow(df_classification), replace=TRUE, prob=c(0.75, 0.25))
train <- df_classification[split,]
test <- df_classification[!split,]

voc_size = 1000

tokenizer <- text_tokenizer(num_words = voc_size)
tokenizer %>% fit_text_tokenizer(df_classification$text)

text_seqs_val_tr <- texts_to_sequences(tokenizer, train$text)
text_seqs_val_te <- texts_to_sequences(tokenizer, test$text)

inputShape = 10

x_train <- text_seqs_val_tr %>% pad_sequences(maxlen = inputShape)
x_test <- text_seqs_val_te %>% pad_sequences(maxlen = inputShape)

y_train <- as.numeric(train$sentiment)
y_test <- as.numeric(test$sentiment)

model <- keras_model_sequential()
model %>%
  layer_embedding(input_dim = voc_size, output_dim = 128, input_length = inputShape) %>%
  bidirectional(layer_lstm(units = 64)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 3, activation = 'softmax')

model %>% compile(
  loss = 'sparse_categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

model %>% fit(
  x_train, y_train,
  batch_size = 32,
  epochs = 4,
  validation_data = list(x_test, y_test)
)