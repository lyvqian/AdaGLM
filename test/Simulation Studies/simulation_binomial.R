library(Matrix)
library(tidyr)
library(microbenchmark)
library(AdaGLM)
library(dplyr)

mse <- function(beta_est, beta_true) {
  mean((beta_est - beta_true)^2)
}

n <- 1000      
p <- 500      
density <- 0.1 
n_replicates <- 100

## binomial

set.seed(815)
run_simulation_binomial <- function(){
  # matrix to store the results
  res_mat <- matrix(NA, nrow=5, ncol=2)
  colnames(res_mat) <- c("MSE", "Time")
  rownames(res_mat) <- c("adam", "adagrad","adadelta", "adasmooth", "glm")
  
  # Simulate sparse X
  X <- rsparsematrix(n, p, density = density)
  
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
  beta_glm <- summary(glm(y~X_dense[,2:ncol(X_dense)], family = binomial()))$coef[,1]
  
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
  select(replicate, name, value) %>%
  pivot_wider(names_from = name, values_from = value)

save(wide_df_binomial, file="/home/lyqian/BIOSTAT815/res_binomial.Rda")

# Gaussian

# run_simulation_gaussian <- function(){
#   # matrix to store the results
#   res_mat <- matrix(NA, nrow=5, ncol=2)
#   colnames(res_mat) <- c("MSE", "Time")
#   rownames(res_mat) <- c("adam", "adagrad","adadelta", "adasmooth", "glm")
#   
#   # Simulate sparse X
#   X <- rsparsematrix(n, p, density = density)
#   
#   beta_true <- rnorm(p)
#   #beta[sample(1:p, size = sparsity * p)] <- 0  # make it 90% sparse
#   
#   mu <- X %*% beta_true
#   
#   y <- as.vector(mu + rnorm(n, mean = 0, sd = 1))
#   
#   X_dense <- as.matrix(X)
#   
#   # Fit GLM
#   family = "gaussian_identity"
#   bench <- suppressWarnings(microbenchmark(
#     beta_adam <- adaglm(X_dense,y,fam_link = family, optimizer = "ADAM"),
#     beta_adagrad <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaGrad"),
#     beta_adadelta <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaDelta"),
#     beta_adasmooth <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaSmooth"),
#     beta_glm <- summary(glm(y~X_dense - 1, family = gaussian()))$coef[,1],
#     times = 1L
#   ))
#   
#   beta_adam <- adaglm(X_dense,y,fam_link = family, optimizer = "ADAM")
#   beta_adagrad <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaGrad")
#   beta_adadelta <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaDelta")
#   beta_adasmooth <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaSmooth")
#   beta_glm <- summary(glm(y~X_dense-1, family = gaussian()))$coef[,1]
#   
#   res_mat[,1] <- c(mse(beta_adam, beta_true), mse(beta_adagrad, beta_true), mse(beta_adadelta, beta_true), mse(beta_adasmooth, beta_true), mse(beta_glm, beta_true))
#   res_mat[,2] <- summary(bench)$median
#   
#   return(res_mat)
# }
# 
# results_gaussian <- do.call(rbind, lapply(1:n_replicates, function(i) run_simulation_gaussian()))
# 
# df_gaussian <- as.data.frame(results_gaussian)
# df_gaussian$method=rownames(results_gaussian)
# df_gaussian$replicate=rep(1:n_replicates, each = 5)
# rownames(df_gaussian) <- NULL
# 
# wide_df_gaussian <- df_gaussian %>%
#   pivot_longer(cols = c(MSE, Time), names_to = "metric", values_to = "value") %>%
#   mutate(name = paste(method, metric, sep = "_")) %>%
#   select(replicate, name, value) %>%
#   pivot_wider(names_from = name, values_from = value)
# 
# save(wide_df_gaussian, file="/home/lyqian/BIOSTAT815/res_gaussian.Rda")
# 
# 
# # Poisson
# 
# run_simulation_poisson <- function(){
#   # matrix to store the results
#   res_mat <- matrix(NA, nrow=5, ncol=2)
#   colnames(res_mat) <- c("MSE", "Time")
#   rownames(res_mat) <- c("adam", "adagrad","adadelta", "adasmooth", "glm")
#   
#   # Simulate sparse X
#   X <- rsparsematrix(n, p, density = density)
#   
#   beta_true <- rnorm(p)
#   #beta[sample(1:p, size = sparsity * p)] <- 0  # make it 90% sparse
#   
#   eta <- X %*% beta_true
#   
#   lambda <- exp(eta)
#   
#   y <- rpois(n, as.vector(lambda))
#   
#   X_dense <- as.matrix(X)
#   
#   # Fit GLM
#   family = "poisson_log"
#   bench <- suppressWarnings(microbenchmark(
#     beta_adam <- adaglm(X_dense,y,fam_link = family, optimizer = "ADAM"),
#     beta_adagrad <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaGrad"),
#     beta_adadelta <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaDelta"),
#     beta_adasmooth <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaSmooth"),
#     beta_glm <- summary(glm(y~X_dense - 1, family = poisson()))$coef[,1],
#     times = 1L
#   ))
#   
#   beta_adam <- adaglm(X_dense,y,fam_link = family, optimizer = "ADAM")
#   beta_adagrad <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaGrad")
#   beta_adadelta <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaDelta")
#   beta_adasmooth <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaSmooth")
#   beta_glm <- summary(glm(y~X_dense-1, family = poisson()))$coef[,1]
#   
#   res_mat[,1] <- c(mse(beta_adam, beta_true), mse(beta_adagrad, beta_true), mse(beta_adadelta, beta_true), mse(beta_adasmooth, beta_true), mse(beta_glm, beta_true))
#   res_mat[,2] <- summary(bench)$median
#   
#   return(res_mat)
# }
# 
# results_poisson <- do.call(rbind, lapply(1:n_replicates, function(i) run_simulation_poisson()))
# 
# df_poisson <- as.data.frame(results_poisson)
# df_poisson$method=rownames(results_poisson)
# df_poisson$replicate=rep(1:n_replicates, each = 5)
# rownames(df_poisson) <- NULL
# 
# wide_df_poisson <- df_poisson %>%
#   pivot_longer(cols = c(MSE, Time), names_to = "metric", values_to = "value") %>%
#   mutate(name = paste(method, metric, sep = "_")) %>%
#   select(replicate, name, value) %>%
#   pivot_wider(names_from = name, values_from = value)
# 
# save(wide_df_poisson, file="/home/lyqian/BIOSTAT815/res_poisson.Rda")
# 

# Gamma

# run_simulation_gamma <- function(){
#   # matrix to store the results
#   res_mat <- matrix(NA, nrow=5, ncol=2)
#   colnames(res_mat) <- c("MSE", "Time")
#   rownames(res_mat) <- c("adam", "adagrad","adadelta", "adasmooth", "glm")
#   
#   phi=1
#   
#   # Simulate sparse X
#   X <- rsparsematrix(n, p, density = density)
#   
#   beta_true <- rnorm(p)
#   #beta[sample(1:p, size = sparsity * p)] <- 0  # make it 90% sparse
#   
#   eta <- X %*% beta_true
#   
#   mu <- as.vector(exp(eta))
#   
#   y <- rgamma(n, shape = 1/phi, scale = mu * phi)
#   
#   X_dense <- as.matrix(X)
#   
#   # Fit GLM
#   family = "Gamma_log"
#   bench <- suppressWarnings(microbenchmark(
#     beta_adam <- adaglm(X_dense,y,fam_link = family, optimizer = "ADAM"),
#     beta_adagrad <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaGrad"),
#     beta_adadelta <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaDelta"),
#     beta_adasmooth <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaSmooth"),
#     beta_glm <- summary(glm(y~X_dense - 1, family = Gamma(link="log")))$coef[,1],
#     times = 1L
#   ))
#   
#   beta_adam <- adaglm(X_dense,y,fam_link = family, optimizer = "ADAM")
#   beta_adagrad <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaGrad")
#   beta_adadelta <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaDelta")
#   beta_adasmooth <- adaglm(X_dense,y,fam_link = family, optimizer = "AdaSmooth")
#   beta_glm <- summary(glm(y~X_dense-1, family = Gamma(link = "log")))$coef[,1]
#   
#   res_mat[,1] <- c(mse(beta_adam, beta_true), mse(beta_adagrad, beta_true), mse(beta_adadelta, beta_true), mse(beta_adasmooth, beta_true), mse(beta_glm, beta_true))
#   res_mat[,2] <- summary(bench)$median
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
#   select(replicate, name, value) %>%
#   pivot_wider(names_from = name, values_from = value)
# 
# save(wide_df_gamma, file="/home/lyqian/BIOSTAT815/res_gamma.Rda")
# 
