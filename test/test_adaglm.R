library(AdaptiveLearningRate)

n <- 1000    
p <- 50

mse <- function(beta_est, beta_true) {
  mean((beta_est - beta_true)^2)
}

## binomial_logit

X <- cbind(matrix(rnorm(n * p), n, p))
true_beta <- c(rnorm(p))  
prob <- 1 / (1 + exp(-X %*% true_beta))
y <- rbinom(n, 1, prob)

beta_adam <- adaglm(X,y,fam_link = "binomial_logit", optimizer = "ADAM")
mse_adam <- mse(beta_adam$coefficients, true_beta)

beta_adagrad <- adaglm(X,y,fam_link = "binomial_logit", optimizer = "AdaGrad")
mse_adagrad <- mse(beta_adagrad$coefficients, true_beta)

beta_adadelta <- adaglm(X,y,fam_link = "binomial_logit", optimizer = "AdaDelta")
mse_adadelta <- mse(beta_adadelta$coefficients, true_beta)

beta_adasmooth <- adaglm(X,y,fam_link = "binomial_logit", optimizer = "AdaSmooth")
mse_adasmooth <- mse(beta_adasmooth$coefficients, true_beta)

beta_glm <- as.numeric(summary(glm(y~ -1 + X, family = binomial()))$coef[,1])
mse_glm <- mse(beta_glm, true_beta)

mse_mat <- cbind(mse_adam, mse_adagrad, mse_adadelta, mse_adasmooth, mse_glm)
colnames(mse_mat) <- c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")
mse_mat

## poisson_log

X <- matrix(rnorm(n * p), nrow = n, ncol = p)
true_beta <- c(runif(5, -0.5, 0.5), rep(0, p - 5))
eta <- X %*% true_beta
mu <- exp(eta)
y <- rpois(n, lambda = mu)

beta_adam <- adaglm(X,y,fam_link = "poisson_log", optimizer = "ADAM")
mse_adam <- mse(beta_adam$coefficients, true_beta)

beta_adagrad <- adaglm(X,y,fam_link = "poisson_log", optimizer = "AdaGrad")
mse_adagrad <- mse(beta_adagrad$coefficients, true_beta)

beta_adadelta <- adaglm(X,y,fam_link = "poisson_log", optimizer = "AdaDelta")
mse_adadelta <- mse(beta_adadelta$coefficients, true_beta)

beta_adasmooth <- adaglm(X,y,fam_link = "poisson_log", optimizer = "AdaSmooth")
mse_adasmooth <- mse(beta_adasmooth$coefficients, true_beta)

beta_glm <- as.numeric(summary(glm(y~ -1 + X, family = poisson()))$coef[,1])
mse_glm <- mse(beta_glm, true_beta)

mse_mat <- cbind(mse_adam, mse_adagrad, mse_adadelta, mse_adasmooth, mse_glm)
colnames(mse_mat) <- c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")
mse_mat

## gaussian_identity

X <- matrix(rnorm(n * p), nrow = n, ncol = p)
true_beta <- c(runif(5, -2, 2), rep(0, p - 5))
eta <- X %*% true_beta
sigma <- 1.0
y <- eta + rnorm(n, mean = 0, sd = sigma)

beta_adam <- adaglm(X,y,fam_link = "gaussian_identity", optimizer = "ADAM")
mse_adam <- mse(beta_adam$coefficients, true_beta)

beta_adagrad <- adaglm(X,y,fam_link = "gaussian_identity", optimizer = "AdaGrad")
mse_adagrad <- mse(beta_adagrad$coefficients, true_beta)

beta_adadelta <- adaglm(X,y,fam_link = "gaussian_identity", optimizer = "AdaDelta")
mse_adadelta <- mse(beta_adadelta$coefficients, true_beta)

beta_adasmooth <- adaglm(X,y,fam_link = "gaussian_identity", optimizer = "AdaSmooth")
mse_adasmooth <- mse(beta_adasmooth$coefficients, true_beta)

beta_glm <- as.numeric(summary(glm(y~ -1 + X, family = gaussian()))$coef[,1])
mse_glm <- mse(beta_glm, true_beta)

mse_mat <- cbind(mse_adam, mse_adagrad, mse_adadelta, mse_adasmooth, mse_glm)
colnames(mse_mat) <- c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")
mse_mat

## Gamma_inverse

X <- cbind(rep(1, n), matrix(rnorm(n * p), nrow = n, ncol = p))
true_beta <- c(runif(6, -1, 1), rep(0, p - 5))
eta <- X %*% true_beta
eta <- pmax(eta, 1e-3)
mu <- 1 / eta
lambda <- 2
y <- rgamma(n, shape = lambda, scale = mu / lambda)

beta_adam <- adaglm(X,y,fam_link = "Gamma_log", optimizer = "ADAM", alpha = 0.001)
mse_adam <- mse(beta_adam$coefficients, true_beta)

beta_adagrad <- adaglm(X,y,fam_link = "Gamma_log", optimizer = "AdaGrad", alpha = 0.001)
mse_adagrad <- mse(beta_adagrad$coefficients, true_beta)

beta_adadelta <- adaglm(X,y,fam_link = "Gamma_log", optimizer = "AdaDelta", rho = 0.99)
mse_adadelta <- mse(beta_adadelta$coefficients, true_beta)

beta_adasmooth <- adaglm(X,y,fam_link = "Gamma_log", optimizer = "AdaSmooth", alpha = 0.0001)
mse_adasmooth <- mse(beta_adasmooth$coefficients, true_beta)

gamma_inv_fit <- glm(y ~ X[,2:51], family = Gamma(link = "log"))
beta_glm <- coef(gamma_inv_fit)
mse_glm <- mean((beta_glm - true_beta)^2)

mse_mat <- cbind(mse_adam, mse_adagrad, mse_adadelta, mse_adasmooth, mse_glm)
colnames(mse_mat) <- c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")
mse_mat


