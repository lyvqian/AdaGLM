library(Rcpp)
library(RcppArmadillo)
library(microbenchmark)
library(pROC)

set.seed(123)

n_replicates <- 100
n <- 1000    
p <- 50  

mse <- function(beta_est, beta_true) {
  mean((beta_est - beta_true)^2)
}

run_simulation <- function() {
  X <- cbind(1, matrix(rnorm(n * p), n, p))
  true_beta <- c(1, rnorm(p))  
  prob <- 1 / (1 + exp(-X %*% true_beta))
  y <- rbinom(n, 1, prob)
  
  beta_adagrad <- adagrad_logistic(X, y)
  beta_adam <- adam_logistic(X, y)
  fit_glm <- glm(y ~ X[, -1], family = binomial)
  beta_glm <- coef(fit_glm)
  
  mse_adagrad <- mse(beta_adagrad, true_beta)
  mse_adam <- mse(beta_adam, true_beta)
  mse_glm <- mse(beta_glm, true_beta)
  
  preds_adagrad <- 1 / (1 + exp(-X %*% beta_adagrad))
  preds_adam <- 1 / (1 + exp(-X %*% beta_adam))
  preds_glm <- predict(fit_glm, type = "response")
  
  acc_adagrad <- mean(ifelse(preds_adagrad > 0.5, 1, 0) == y)
  acc_adam <- mean(ifelse(preds_adam > 0.5, 1, 0) == y)
  acc_glm <- mean(ifelse(preds_glm > 0.5, 1, 0) == y)
  
  auc_adagrad <- auc(y, as.numeric(preds_adagrad))
  auc_adam <- auc(y, as.numeric(preds_adam))
  auc_glm <- auc(y, as.numeric(preds_glm))
  
  bench <- suppressWarnings(microbenchmark(
    glm = glm(y ~ X[, -1], family = binomial),
    adagrad = adagrad_logistic(X, y),
    adam = adam_logistic(X, y),
    times = 1L
  ))
  
  exec_time_glm <- summary(bench)$median[1]
  exec_time_adagrad <- summary(bench)$median[2]
  exec_time_adam <- summary(bench)$median[3]
  
  return(data.frame(mse_adagrad, mse_adam, mse_glm, acc_adagrad, acc_adam, acc_glm, auc_adagrad, auc_adam, auc_glm, exec_time_adagrad, exec_time_adam, exec_time_glm))
}

results <- do.call(rbind, lapply(1:n_replicates, function(i) run_simulation()))

summary(results)


boxplot(results$mse_adagrad, results$mse_adam, results$mse_glm, 
        names = c("AdaGrad", "Adam", "GLM"),
        main = "MSE (beta)",
        ylab = "Mean Squared Error")

boxplot(results$acc_adagrad, results$acc_adam, results$acc_glm, 
        names = c("AdaGrad", "Adam", "GLM"),
        main = "Prediction Accuracy",
        ylab = "Classification Accuracy")

boxplot(results$exec_time_adagrad, results$exec_time_adam, results$exec_time_glm, 
        names = c("AdaGrad", "Adam", "GLM"),
        main = "Execution Time",
        ylab = "Time (ms)")




