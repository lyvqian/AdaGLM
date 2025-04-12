library(Matrix)
library(tidyr)
library(microbenchmark)
library(AdaGLM)
library(dplyr)

mse <- function(beta_est, beta_true) {
  mean((beta_est - beta_true)^2)
}

n <- 100      
p <- 5    
n_replicates <- 100

## binomial

set.seed(123)
run_simulation_binomial <- function(){
  # matrix to store the results
  res_mat <- matrix(NA, nrow=5, ncol=2)
  colnames(res_mat) <- c("MSE", "Time")
  rownames(res_mat) <- c("adam", "adagrad","adadelta", "adasmooth", "glm")
  
  # Simulate sparse X
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  
  beta_true <- rnorm(p)
  #beta[sample(1:p, size = sparsity * p)] <- 0  # make it 90% sparse
  
  eta <- X %*% beta_true
  prob <- 1 / (1 + exp(-eta))
  
  y <- rbinom(n, 1, as.numeric(prob))
  
  X_dense <- as.matrix(X)
  
  # Fit GLM
  family = "binomial_logit"
  bench <- suppressWarnings(microbenchmark(
    beta_adam <- adaglm(X_dense,y,fam_link = family, optimizer = "ADAM", alpha=0.001),
    beta_adagrad <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaGrad", alpha=0.1),
    beta_adadelta <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaDelta"),
    beta_adasmooth <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaSmooth", alpha=0.001),
    beta_glm <- summary(glm(y~X_dense - 1, family = binomial()))$coef[,1],
    times = 1L
  ))
  
  beta_adam <- adaglm(X_dense,y,fam_link = family, optimizer = "ADAM", alpha=0.001)
  beta_adagrad <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaGrad", alpha=0.1)
  beta_adadelta <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaDelta")
  beta_adasmooth <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaSmooth", alpha=0.001)
  beta_glm <- summary(glm(y~X_dense-1, family = binomial()))$coef[,1]
  
  res_mat[,1] <- c(mse(beta_adam, beta_true), mse(beta_adagrad, beta_true), mse(beta_adadelta, beta_true), mse(beta_adasmooth, beta_true), mse(beta_glm, beta_true))
  res_mat[,2] <- summary(bench)$median
  
  return(res_mat)
}

results_binomial <- do.call(rbind, lapply(1:n_replicates, function(i) run_simulation_binomial()))

df_binomial <- as.data.frame(results_binomial)
df_binomial$method=rownames(results_binomial)
df_binomial$replicate=rep(1:n_replicates, each = 5)
rownames(df_binomial) <- NULL

wide_df_binomial <- df_binomial %>%
  pivot_longer(cols = c(MSE, Time), names_to = "metric", values_to = "value") %>%
  mutate(name = paste(method, metric, sep = "_")) %>%
  dplyr::select(replicate, name, value) %>%
  pivot_wider(names_from = name, values_from = value)

save(wide_df_binomial, file="./test/res_binomial_small.Rda")

# Gaussian

run_simulation_gaussian <- function(){
  # matrix to store the results
  res_mat <- matrix(NA, nrow=5, ncol=2)
  colnames(res_mat) <- c("MSE", "Time")
  rownames(res_mat) <- c("adam", "adagrad","adadelta", "adasmooth", "glm")
  
  # Simulate sparse X
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  
  beta_true <- rnorm(p)
  #beta[sample(1:p, size = sparsity * p)] <- 0  # make it 90% sparse
  
  mu <- X %*% beta_true
  
  y <- as.vector(mu + rnorm(n, mean = 0, sd = 1))
  
  X_dense <- as.matrix(X)
  
  # Fit GLM
  family = "gaussian_identity"
  bench <- suppressWarnings(microbenchmark(
    beta_adam <- adaglm(X_dense,y,fam_link = family, optimizer = "ADAM", alpha=0.001),
    beta_adagrad <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaGrad", alpha = 0.1),
    beta_adadelta <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaDelta"),
    beta_adasmooth <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaSmooth", alpha=0.001),
    beta_glm <- summary(glm(y~X_dense - 1, family = gaussian()))$coef[,1],
    times = 1L
  ))
  
  beta_adam <- adaglm(X_dense,y,fam_link = family, optimizer = "ADAM", alpha=0.001)
  beta_adagrad <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaGrad", alpha = 0.1)
  beta_adadelta <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaDelta")
  beta_adasmooth <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaSmooth", alpha=0.001)
  beta_glm <- summary(glm(y~X_dense-1, family = gaussian()))$coef[,1]
  
  res_mat[,1] <- c(mse(beta_adam, beta_true), mse(beta_adagrad, beta_true), mse(beta_adadelta, beta_true), mse(beta_adasmooth, beta_true), mse(beta_glm, beta_true))
  res_mat[,2] <- summary(bench)$median
  
  return(res_mat)
}

results_gaussian <- do.call(rbind, lapply(1:n_replicates, function(i) run_simulation_gaussian()))

df_gaussian <- as.data.frame(results_gaussian)
df_gaussian$method=rownames(results_gaussian)
df_gaussian$replicate=rep(1:n_replicates, each = 5)
rownames(df_gaussian) <- NULL

wide_df_gaussian <- df_gaussian %>%
  pivot_longer(cols = c(MSE, Time), names_to = "metric", values_to = "value") %>%
  mutate(name = paste(method, metric, sep = "_")) %>%
  dplyr::select(replicate, name, value) %>%
  pivot_wider(names_from = name, values_from = value)

save(wide_df_gaussian, file="./test/res_gaussian_small.Rda")


# Poisson

run_simulation_poisson <- function(){
  # matrix to store the results
  res_mat <- matrix(NA, nrow=5, ncol=2)
  colnames(res_mat) <- c("MSE", "Time")
  rownames(res_mat) <- c("adam", "adagrad","adadelta", "adasmooth", "glm")
  
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  
  beta_true <- rnorm(p)
  #beta[sample(1:p, size = sparsity * p)] <- 0  # make it 90% sparse
  
  eta <- X %*% beta_true
  
  lambda <- exp(eta)
  
  y <- rpois(n, as.vector(lambda))
  
  X_dense <- as.matrix(X)
  
  # Fit GLM
  family = "poisson_log"
  bench <- suppressWarnings(microbenchmark(
    beta_adam <- adaglm(X_dense,y,fam_link = family, optimizer = "ADAM", alpha=0.001),
    beta_adagrad <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaGrad", alpha=0.1),
    beta_adadelta <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaDelta"),
    beta_adasmooth <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaSmooth", alpha=0.001),
    beta_glm <- summary(glm(y~X_dense - 1, family = poisson()))$coef[,1],
    times = 1L
  ))
  
  beta_adam <- adaglm(X_dense,y,fam_link = family, optimizer = "ADAM", alpha=0.001)
  beta_adagrad <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaGrad", alpha=0.1)
  beta_adadelta <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaDelta")
  beta_adasmooth <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaSmooth", alpha=0.001)
  beta_glm <- summary(glm(y~X_dense-1, family = poisson()))$coef[,1]
  
  res_mat[,1] <- c(mse(beta_adam, beta_true), mse(beta_adagrad, beta_true), mse(beta_adadelta, beta_true), mse(beta_adasmooth, beta_true), mse(beta_glm, beta_true))
  res_mat[,2] <- summary(bench)$median
  
  return(res_mat)
}

results_poisson <- do.call(rbind, lapply(1:n_replicates, function(i) run_simulation_poisson()))

df_poisson <- as.data.frame(results_poisson)
df_poisson$method=rownames(results_poisson)
df_poisson$replicate=rep(1:n_replicates, each = 5)
rownames(df_poisson) <- NULL

wide_df_poisson <- df_poisson %>%
  pivot_longer(cols = c(MSE, Time), names_to = "metric", values_to = "value") %>%
  mutate(name = paste(method, metric, sep = "_")) %>%
  dplyr::select(replicate, name, value) %>%
  pivot_wider(names_from = name, values_from = value)

save(wide_df_poisson, file="./test/res_poisson_small.Rda")

# Gamma

# run_simulation_gamma <- function(){
#   # matrix to store the results
#   res_mat <- matrix(NA, nrow=5, ncol=2)
#   colnames(res_mat) <- c("MSE", "Time")
#   rownames(res_mat) <- c("adam", "adagrad","adadelta", "adasmooth", "glm")
#   
#   phi=0.5
#   
#   X <- matrix(rnorm(n * p) + 1e-8, nrow = n, ncol = p)
#   
#   beta_true <- rnorm(p)
#   #beta[sample(1:p, size = sparsity * p)] <- 0  # make it 90% sparse
#   
#   eta <- X %*% beta_true
#   
#   eta <- pmin(pmax(eta, -5), 5)
#   
#   mu <- as.vector(exp(eta))
#   
#   y <- rgamma(n, shape = 1/phi, scale = mu * phi)
#   
#   y <-  pmax(y, 100)
# 
#   X_dense <- as.matrix(X)
#   
#   # Fit GLM
#   family = "Gamma_log"
#   bench <- suppressWarnings(microbenchmark(
#     beta_adam <- adaglm(X_dense,y,fam_link = family, optimizer = "ADAM"),
#     beta_adagrad <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaGrad", alpha=0.1),
#     beta_adadelta <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaDelta"),
#     beta_adasmooth <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaSmooth"),
#     #beta_glm <- summary(glm(y~X_dense - 1, family = Gamma(link="log")))$coef[,1],
#     times = 1L
#   ))
#   
#   glm_time <- system.time({
#     beta_glm <- tryCatch({
#       coef(glm(
#         y ~ X_dense - 1,
#         family = Gamma(link = "log"),
#         control = glm.control(maxit = 100, epsilon = 1e-8)
#       ))
#     }, error = function(e) {
#       warning("GLM failed: ", conditionMessage(e))
#       rep(NA, p)
#     })
#   })[["elapsed"]]
#   
#   beta_adam <- adaglm(X_dense,y,fam_link = family, optimizer = "ADAM")
#   beta_adagrad <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaGrad", alpha=0.1)
#   beta_adadelta <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaDelta")
#   beta_adasmooth <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaSmooth")
#   beta_glm <- tryCatch({
#     coef(glm(y ~ X_dense - 1, family = Gamma(link = "log")))
#   }, error = function(e) {
#     warning("GLM failed to converge: ", conditionMessage(e))
#     rep(NA, p)
#   })
#   
#   res_mat[,1] <- c(mse(beta_adam, beta_true), mse(beta_adagrad, beta_true), mse(beta_adadelta, beta_true), mse(beta_adasmooth, beta_true), mse(beta_glm, beta_true))
#   res_mat[,2] <- c(summary(bench)$median, glm_time)
#   
#   return(res_mat)
# }
# 
# results_gamma <- do.call(rbind, lapply(1:n_replicates, function(i) run_simulation_gamma()))
# 
# df_gamma <- as.data.frame(results_gamma)
# df_gamma$method=rownames(results_gamma)
# df_gamma$replicate=rep(1:n_replicates, each = 5)
# rownames(df_gamma) <- NULL
# 
# wide_df_gamma <- df_gamma %>%
#   pivot_longer(cols = c(MSE, Time), names_to = "metric", values_to = "value") %>%
#   mutate(name = paste(method, metric, sep = "_")) %>%
#   dplyr::select(replicate, name, value) %>%
#   pivot_wider(names_from = name, values_from = value)
# 
# save(wide_df_gamma, file="./test/res_gamma_small.Rda")
# 
