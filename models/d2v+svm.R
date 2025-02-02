# Load required libraries
library(text2vec) # For text embeddings
library(e1071)    # For SVM
library(caTools)  # For train-test split
library(tidyr)

calcium <- read.csv2("calcium_preprocessed.csv", sep=",")
calcium <- subset(calcium, select =c("title","label_included") )
data <- calcium

# Split data into train and test sets
set.seed(12345) # For reproducibility
split <- sample.split(data$label_included, SplitRatio = 0.7)
train_data <- subset(data, split == TRUE)
test_data <- subset(data, split == FALSE)

# Tokenize the text
prep_fun <- tolower
tokenizer <- word_tokenizer

train_tokens <- train_data$title %>% prep_fun() %>% tokenizer()
test_tokens <- test_data$title %>% prep_fun() %>% tokenizer()

# Create vocabulary and term-co-occurrence matrix
it_train <- itoken(train_tokens, progressbar = FALSE)
vocab <- create_vocabulary(it_train)
vectorizer <- vocab_vectorizer(vocab)

tcm <- create_tcm(it_train, vectorizer, skip_grams_window = 5)

# Fit GloVe model to generate word embeddings
glove <- GlobalVectors$new(rank = 80, x_max = 20)
word_vectors <- glove$fit_transform(tcm, n_iter = 20)

# Create document embeddings by averaging word vectors
create_document_embeddings <- function(tokens, word_vectors) {
  embeddings <- lapply(tokens, function(doc) {
    # Filter valid words that exist in the word_vectors
    valid_words <- doc[doc %in% rownames(word_vectors)]
    
    # If there are valid words, compute the mean embedding
    if (length(valid_words) > 0) {
      return(colMeans(word_vectors[valid_words, , drop = FALSE]))
    } else {
      # If no valid words, return a zero vector
      return(rep(0, ncol(word_vectors)))
    }
  })
  
  # Combine all document embeddings into a single matrix
  do.call(rbind, embeddings)
}



# Compute consistent document embeddings for train and test sets
train_embeddings <- create_document_embeddings(train_tokens, word_vectors)
test_embeddings <- create_document_embeddings(test_tokens, word_vectors)

# Check dimensions of train and test embeddings
cat(sprintf("Train Embedding Dimensions: %d x %d\n", nrow(train_embeddings), ncol(train_embeddings)))
cat(sprintf("Test Embedding Dimensions: %d x %d\n", nrow(test_embeddings), ncol(test_embeddings)))

# Train an SVM classifier
svm_model <- svm(
  x = train_embeddings,
  y = as.factor(train_data$label), # Labels must be factors for classification
  kernel = "linear",              # Linear kernel
  cost = 1,                       # Regularization parameter
  scale = TRUE                    # Scale features
)

# Predict on the test set
test_predictions <- predict(svm_model, newdata = test_embeddings)

# Evaluate the model
conf_matrix <- table(Predicted = test_predictions, Actual = test_data$label)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)

# Print results
cat("Confusion Matrix:\n")
print(conf_matrix)
cat(sprintf("Accuracy: %.2f%%\n", accuracy * 100))
WSS <- 335/(335+30)-0.05
WSS85 <- 335/(335+30)-0.15

virus <- read.csv2("virus_preprocessed.csv", sep=",")
virus <- subset(virus, select =c("titles","label_included") )
data <- virus
WSS95_v <- 744/(744+36)-0.05
WSS85_v <- 744/(744+36)-0.15

depression <- read.csv2("depression_preprocessed.csv", sep=",")
depression <- subset(depression, select =c("titles","label_included") )
data <- depression
WSS95_d <- (488+26)/(488+26+61+23)-0.05
WSS85_d <- (488+26)/(488+26+61+23)-0.15

results <- data.frame(
  Dataset = c("Calcium", "Virus", "Depression"),
  Model = c("SVM", "SVM", "SVM"),
  WSS85 = c(WSS85, WSS85_v, WSS85_d),
  WSS95 = c(WSS, WSS95_v, WSS95_d)
)

# Define the results file path (one folder above)
results_file <- "../results.csv"

# Check if the file exists
if (file.exists(results_file)) {
  # Append to existing CSV file
  write.table(results, file = results_file, append = TRUE, sep = ",", col.names = FALSE, row.names = FALSE)
} else {
  # Create new CSV file with headers
  write.table(results, file = results_file, sep = ",", col.names = TRUE, row.names = FALSE)
}

cat("Results saved to ../results.csv\n")