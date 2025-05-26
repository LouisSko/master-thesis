library(quantreg)

# Simulate data
set.seed(123)
n <- 100000
x <- rnorm(n)
y <- 1 + 2 * x + rnorm(n)

# Fit quantile regression at median (tau = 0.5)
start_time <- Sys.time()
model <- rq(y ~ x, tau = 0.1)
end_time <- Sys.time()

# Summary and timing
print(summary(model))
cat("Time taken (R):", end_time - start_time, "\n")


# now test it with actual predictions

# read predictions and run regression
df <- read.csv("/Users/louisskowronek/Documents/thesis/master-thesis/archive/test-prediction.csv")

# Ensure numeric conversion
for (col in names(df)) {
  df[[col]] <- suppressWarnings(as.numeric(as.character(df[[col]])))
}

# Define quantile levels
a_seq <- seq(0.1, 0.9, by = 0.1)
# results <- list()

# Loop over quantile columns
time_start <- Sys.time()
for (alpha in a_seq) {
  colname <- paste0("X", alpha)
  fml <- as.formula(paste("target ~", colname))
  
  # Fit quantile regression
  model <- rq(fml, tau = alpha, data = df)
  
  # Save result
  # results[[colname]] <- summary(model)
}
time_end <- Sys.time()
print(summary(model))
cat("Time taken (R):", time_end - time_start, "\n")
