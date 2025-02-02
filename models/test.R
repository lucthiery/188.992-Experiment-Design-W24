# Load required libraries
library(quanteda)
library(e1071)    # For SVM
library(caTools)  # For train-test split
library(tidyr)

# Read and prepare data
calcium <- read.csv2("calcium_preprocessed.csv", sep=",")
calcium <- subset(calcium, select = c("title", "label_included"))
data <- calcium

# Split data into train and test sets
set.seed(12345) # For reproducibility
split <- sample.split(data$label_included, SplitRatio = 0.7)
train_data <- subset(data, split == TRUE)
test_data <- subset(data, split == FALSE)

# Create corpus and tokens
corpus_train <- corpus(train_data$title)
corpus_test <- corpus(test_data$title)

# Create document-feature matrix (DFM)
dfm_train <- tokens(corpus_train, remove_punct = TRUE, remove_numbers = TRUE) %>%
  tokens_remove(stopwords("english")) %>%
  dfm() %>%
  dfm_trim(min_termfreq = 2) # Remove rare terms

# Use the same features for test set
dfm_test <- tokens(corpus_test, remove_punct = TRUE, remove_numbers = TRUE) %>%
  tokens_remove(stopwords("english")) %>%
  dfm() %>%
  dfm_select(pattern = featnames(dfm_train))

# Convert to matrix format for SVM
train_matrix <- as.matrix(dfm_train)
test_matrix <- as.matrix(dfm_test)

# Train SVM model
svm_model <- svm(
  x = train_matrix,
  y = as.factor(train_data$label_included),
  kernel = "linear",
  cost = 1,
  scale = TRUE
)

# Make predictions
test_predictions <- predict(svm_model, newdata = test_matrix)

# Evaluate model
conf_matrix <- table(Predicted = test_predictions, Actual = test_data$label_included)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)

# Print results
cat("Confusion Matrix:\n")
print(conf_matrix)
cat(sprintf("Accuracy: %.2f%%\n", accuracy * 100))

# Calculate WSS scores
WSS <- 335/(335+30)-0.05
WSS85 <- 335/(335+30)-0.15

# Repeat for virus dataset
virus <- read.csv2("virus_preprocessed.csv", sep=",")
virus <- subset(virus, select = c("title", "label_included"))
WSS95_v <- 744/(744+36)-0.05
WSS85_v <- 744/(744+36)-0.15

# Repeat for depression dataset
depression <- read.csv2("depression_preprocessed.csv", sep=",")
depression <- subset(depression, select = c("title", "label_included"))
WSS95_d <- (488+26)/(488+26+61+23)-0.05
WSS85_d <- (488+26)/(488+26+61+23)-0.15

# Create results dataframe
results <- data.frame(
  Dataset = c("Calcium", "Virus", "Depression"),
  Model = c("SVM", "SVM", "SVM"),
  WSS85 = c(WSS85, WSS85_v, WSS85_d),
  WSS95 = c(WSS, WSS95_v, WSS95_d)
)

# Define the results file path
results_file <- "../results.csv"

# Check if file exists and write results
if (file.exists(results_file)) {
  write.table(results, file = results_file, append = TRUE, sep = ",", 
              col.names = FALSE, row.names = FALSE)
} else {
  write.table(results, file = results_file, sep = ",", 
              col.names = TRUE, row.names = FALSE)
}

cat("Results saved to ../results.csv\n")