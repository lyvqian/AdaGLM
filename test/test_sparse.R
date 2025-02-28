library(Rcpp)
library(RcppArmadillo)
library(microbenchmark)
library(pROC)

set.seed(123)

n <- 1000    
p <- 500     
sparsity <- 0.9  # 90% of elements are zero

n_replicates <- 20

mse <- function(beta_est, beta_true) {
  mean((beta_est - beta_true)^2)
}

run_simulation <- function() {
  X <- matrix(rnorm(n * p), n, p)
  X[sample(length(X), size = floor(sparsity * length(X)))] <- 0
  X <- cbind(1, X)
  
  beta_true <- c(1, rep(0, p))
  non_zero_indices <- sample(2:(p + 1), 10)
  beta_true[non_zero_indices] <- rnorm(10)
  
  prob <- 1 / (1 + exp(-X %*% beta_true))
  y <- rbinom(n, 1, prob)
  
  beta_adagrad <- adagrad_logistic(X, y)
  beta_adam <- adam_logistic(X, y)
  beta_adasmooth <- adasmooth_logistic(X, y)
  beta_adadelta <- adadelta_logistic(X, y)
  beta_irwls <- irwls_logistic(X, y)
  # fit_glm <- glm(y ~ X[, -1], family = binomial)
  # beta_glm <- coef(fit_glm)
  
  mse_adagrad <- mse(beta_adagrad, beta_true)
  mse_adam <- mse(beta_adam, beta_true)
  mse_adasmooth <- mse(beta_adasmooth, beta_true)
  mse_adadelta <- mse(beta_adadelta, beta_true)
  mse_irwls <- mse(beta_irwls, beta_true)
  #mse_glm <- mse(beta_glm, beta_true)
  
  preds_adagrad <- 1 / (1 + exp(-X %*% beta_adagrad))
  preds_adam <- 1 / (1 + exp(-X %*% beta_adam))
  preds_adasmooth <- 1 / (1 + exp(-X %*% beta_adasmooth))
  preds_adadelta <- 1 / (1 + exp(-X %*% beta_adadelta))
  preds_irwls <- 1 / (1 + exp(-X %*% beta_irwls))
  #preds_glm <- predict(fit_glm, type = "response")
  
  acc_adagrad <- mean(ifelse(preds_adagrad > 0.5, 1, 0) == y)
  acc_adam <- mean(ifelse(preds_adam > 0.5, 1, 0) == y)
  acc_adasmooth <- mean(ifelse(preds_adasmooth > 0.5, 1, 0) == y)
  acc_adadelta <- mean(ifelse(preds_adadelta > 0.5, 1, 0) == y)
  acc_irwls <- mean(ifelse(preds_irwls > 0.5, 1, 0) == y)
  #acc_glm <- mean(ifelse(preds_glm > 0.5, 1, 0) == y)
  
  # auc_adagrad <- auc(y, preds_adagrad)
  # auc_adam <- auc(y, preds_adam)
  # auc_glm <- auc(y, preds_glm)
  
  bench <- suppressWarnings(microbenchmark(
    #glm = glm(y ~ X[, -1], family = binomial),
    irwls = irwls_logistic(X, y),
    adagrad = adagrad_logistic(X, y),
    adam = adam_logistic(X, y),
    adasmooth = adasmooth_logistic(X, y),
    adadelta = adadelta_logistic(X, y),
    times = 1L
  ))
  
  exec_time_irwls <- summary(bench)$median[1]
  #exec_time_glm <- summary(bench)$median[1]
  exec_time_adagrad <- summary(bench)$median[2]
  exec_time_adam <- summary(bench)$median[3]
  exec_time_adasmooth <- summary(bench)$median[4]
  exec_time_adadelta <- summary(bench)$median[5]
  
  return(data.frame(mse_adagrad, mse_adam, mse_irwls, mse_adadelta, mse_adasmooth,
                    acc_adagrad, acc_adam, acc_irwls, acc_adadelta, acc_adasmooth,
                    #auc_adagrad, auc_adam, auc_glm, 
                    exec_time_adagrad, exec_time_adam, exec_time_irwls, exec_time_adadelta, exec_time_adasmooth))
}

results <- do.call(rbind, lapply(1:n_replicates, function(i) run_simulation()))

summary(results)

boxplot(results$mse_adagrad, results$mse_adam, results$mse_adasmooth, results$mse_adadelta, results$mse_irwls, 
        names = c("AdaGrad", "Adam", "AdaSmooth", "AdaDelta", "IRWLS"),
        main = "MSE (beta)",
        ylab = "Mean Squared Error")

boxplot(results$acc_adagrad, results$acc_adam, results$acc_adasmooth, results$acc_adadelta, results$acc_irwls, 
        names = c("AdaGrad", "Adam", "AdaSmooth", "AdaDelta", "IRWLS"),
        main = "Prediction Accuracy",
        ylab = "Classification Accuracy")

boxplot(results$exec_time_adagrad, results$exec_time_adam, results$exec_time_adasmooth, results$exec_time_adadelta, results$exec_time_irwls, 
        names = c("AdaGrad", "Adam", "AdaSmooth", "AdaDelta", "IRWLS"),
        main = "Execution Time",
        ylab = "Time (ms)")


##############################################################

n_replicates <- 100

n <- 1000    
p <- 50     
sparsity <- 0.9  # 90% of elements are zero

run_simulation <- function() {
  X <- matrix(rnorm(n * p), n, p)
  X[sample(length(X), size = floor(sparsity * length(X)))] <- 0
  X <- cbind(1, X)
  
  beta_true <- c(1, rep(0, p))
  non_zero_indices <- sample(2:(p + 1), 10)
  beta_true[non_zero_indices] <- rnorm(10)
  
  prob <- 1 / (1 + exp(-X %*% beta_true))
  y <- rbinom(n, 1, prob)
  
  beta_adagrad <- adagrad_logistic(X, y)
  beta_adam <- adam_logistic(X, y)
  beta_adasmooth <- adasmooth_logistic(X, y)
  beta_adadelta <- adadelta_logistic(X, y)
  beta_irwls <- irwls_logistic(X, y)
  # fit_glm <- glm(y ~ X[, -1], family = binomial)
  # beta_glm <- coef(fit_glm)
  
  mse_adagrad <- mse(beta_adagrad, beta_true)
  mse_adam <- mse(beta_adam, beta_true)
  mse_adasmooth <- mse(beta_adasmooth, beta_true)
  mse_adadelta <- mse(beta_adadelta, beta_true)
  mse_irwls <- mse(beta_irwls, beta_true)
  #mse_glm <- mse(beta_glm, beta_true)
  
  preds_adagrad <- 1 / (1 + exp(-X %*% beta_adagrad))
  preds_adam <- 1 / (1 + exp(-X %*% beta_adam))
  preds_adasmooth <- 1 / (1 + exp(-X %*% beta_adasmooth))
  preds_adadelta <- 1 / (1 + exp(-X %*% beta_adadelta))
  preds_irwls <- 1 / (1 + exp(-X %*% beta_irwls))
  #preds_glm <- predict(fit_glm, type = "response")
  
  acc_adagrad <- mean(ifelse(preds_adagrad > 0.5, 1, 0) == y)
  acc_adam <- mean(ifelse(preds_adam > 0.5, 1, 0) == y)
  acc_adasmooth <- mean(ifelse(preds_adasmooth > 0.5, 1, 0) == y)
  acc_adadelta <- mean(ifelse(preds_adadelta > 0.5, 1, 0) == y)
  acc_irwls <- mean(ifelse(preds_irwls > 0.5, 1, 0) == y)
  #acc_glm <- mean(ifelse(preds_glm > 0.5, 1, 0) == y)
  
  # auc_adagrad <- auc(y, preds_adagrad)
  # auc_adam <- auc(y, preds_adam)
  # auc_glm <- auc(y, preds_glm)
  
  bench <- suppressWarnings(microbenchmark(
    #glm = glm(y ~ X[, -1], family = binomial),
    irwls = irwls_logistic(X, y),
    adagrad = adagrad_logistic(X, y),
    adam = adam_logistic(X, y),
    adasmooth = adasmooth_logistic(X, y),
    adadelta = adadelta_logistic(X, y),
    times = 1L
  ))
  
  exec_time_irwls <- summary(bench)$median[1]
  #exec_time_glm <- summary(bench)$median[1]
  exec_time_adagrad <- summary(bench)$median[2]
  exec_time_adam <- summary(bench)$median[3]
  exec_time_adasmooth <- summary(bench)$median[4]
  exec_time_adadelta <- summary(bench)$median[5]
  
  return(data.frame(mse_adagrad, mse_adam, mse_irwls, mse_adadelta, mse_adasmooth,
                    acc_adagrad, acc_adam, acc_irwls, acc_adadelta, acc_adasmooth,
                    #auc_adagrad, auc_adam, auc_glm, 
                    exec_time_adagrad, exec_time_adam, exec_time_irwls, exec_time_adadelta, exec_time_adasmooth))
}

results <- do.call(rbind, lapply(1:n_replicates, function(i) run_simulation()))

summary(results)

boxplot(results$mse_adagrad, results$mse_adam, results$mse_adasmooth, results$mse_adadelta, results$mse_irwls, 
        names = c("AdaGrad", "Adam", "AdaSmooth", "AdaDelta", "IRWLS"),
        main = "MSE (beta)",
        ylab = "Mean Squared Error")

boxplot(results$acc_adagrad, results$acc_adam, results$acc_adasmooth, results$acc_adadelta, results$acc_irwls, 
        names = c("AdaGrad", "Adam", "AdaSmooth", "AdaDelta", "IRWLS"),
        main = "Prediction Accuracy",
        ylab = "Classification Accuracy")

boxplot(results$exec_time_adagrad, results$exec_time_adam, results$exec_time_adasmooth, results$exec_time_adadelta, results$exec_time_irwls, 
        names = c("AdaGrad", "Adam", "AdaSmooth", "AdaDelta", "IRWLS"),
        main = "Execution Time",
        ylab = "Time (ms)")


