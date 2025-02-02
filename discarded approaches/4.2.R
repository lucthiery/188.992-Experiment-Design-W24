#install.packages("keras")
#install.packages("tensorflow")

library(tensorflow)
library(keras)

data <- read.csv2("representation_embeddings.csv", sep=",")
dim(data)
dim_input <- 1218
dim_latent <- 10
dim_hidden <- 128

n <- 768
train <- sample(1:n, round(n*2/3))
test <- (1:n)[-train]
x_train <- df_normalized[train,]
x_test <- df_normalized[test,]
x_train <- array_reshape(x_train, c(nrow(x_train), dim_input))
x_test <- array_reshape(x_test, c(nrow(x_test), dim_input))

df <- lapply(data$doc_embedding, function(x) {
  x <- gsub("\\[|\\]", "", x)  # Remove square brackets
  numeric_vector <- as.numeric(unlist(strsplit(x, " ")))  # Split by comma and convert to numeric
  numeric_vector[!is.na(numeric_vector)]
})

df_1 <- as.data.frame(df, col.names = "")

normalize_min_max <- function(x) {
  (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
}

df_normalized <- as.data.frame(lapply(df_1, function(column) {
  if (is.numeric(column)) normalize_min_max(column) else column
}))


# encoder
  # data into estimates of mean and variance of latent space
  # layer used as an entry point into a graph
input <- layer_input(shape=dim_input, name="encoder_input")
x <- layer_dense(input, dim_hidden, activation = "relu", name = "intermediate_layer")
z_mean <- layer_dense(x, dim_latent, name = "z_mean")
z_log_var <- layer_dense(x, dim_latent, name = "z_log_var")

encoder <- keras_model(input, list(z_mean, z_log_var), name="encoder")
encoder

# then sample from this distribution
  #'reparameterization'
sampling <- function(args) {
  z_mean <- args[[1]]
  z_log_var <- args[[2]]
  epsilon <- k_random_normal(shape = k_shape(z_mean))
  z_mean + k_exp(0.5 * z_log_var) * epsilon
}
z <- layer_lambda(list(z_mean, z_log_var), sampling, name = "z")

layer_sampler <- new_layer_class(
  classname = "Sampler",
  call = function(z_mean, z_log_var) {
    epsilon <- tf$random$normal(shape = tf$shape(z_mean))
    z_mean + exp(0.5 * z_log_var) * epsilon }
)

# decoder
decoder_input <- layer_input(shape = dim_latent, name = "decoder_input")
x_decoded <- layer_dense(decoder_input,dim_hidden, activation = "relu")
outputs <- layer_dense(x_decoded, dim_input, activation = "sigmoid", name = "decoder_output")

decoder <- keras_model(decoder_input, outputs, name = "decoder")
decoder

# VAE model ----
 # combine encoder and decoder into a model
#vae <- keras_model(input, decoded)

## loss function ----
vae_loss <- function(y_true, y_pred) {
  reconstruction_loss <- loss_binary_crossentropy(y_true, y_pred) * dim_input
  kl_loss <- -0.5 * k_sum(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1)
  k_mean(reconstruction_loss + kl_loss)
}

vae_1 <- model_vae(encoder, decoder)
vae_1 %>% compile(optimizer=optimizer_adam())
vae_1 %>% fit(x_train, epochs=20, shuffle=T)

model_vae <- new_model_class(
  classname = "VAE",
  
  initialize = function(encoder, decoder, ...) {
    super$initialize(...)
    self$encoder <- encoder
    self$decoder <- decoder
    self$sampler <- layer_sampler()
    self$total_loss_tracker <- metric_mean(name = "total_loss")
    self$reconstruction_loss_tracker <- metric_mean(name = "reconstruction_loss")
    self$kl_loss_tracker <- metric_mean(name = "kl_loss")
  },
  
  metrics = mark_active(function() {
    list(
      self$total_loss_tracker,
      self$reconstruction_loss_tracker,
      self$kl_loss_tracker
    )
  }),
  
  train_step = function(data) {
    with(tf$GradientTape() %as% tape, {
      
      c(z_mean, z_log_var) %<-% self$encoder(data)
      z <- self$sampler(z_mean, z_log_var)
      
      reconstruction <- decoder(z)
      reconstruction_loss <-
        loss_binary_crossentropy(data, reconstruction) %>%
        sum(axis = c(1)) %>%
        mean()
      
      kl_loss <- -0.5 * tf$reduce_mean(1 + z_log_var - tf$square(z_mean) - tf$exp(z_log_var))
      total_loss <- reconstruction_loss + mean(kl_loss)
    })
    
    tf$print("Reconstruction Loss:", reconstruction_loss)
    tf$print("KL Loss:", kl_loss)
    tf$print("Total Loss:", total_loss)
    
    grads <- tape$gradient(total_loss, self$trainable_weights)
    self$optimizer$apply_gradients(zip_lists(grads, self$trainable_weights))
    
    self$total_loss_tracker$update_state(total_loss)
    self$reconstruction_loss_tracker$update_state(reconstruction_loss)
    self$kl_loss_tracker$update_state(kl_loss)
    
    list(total_loss = self$total_loss_tracker$result(),
         reconstruction_loss = self$reconstruction_loss_tracker$result(),
         kl_loss = self$kl_loss_tracker$result())
  }
)

