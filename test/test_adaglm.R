n <- 1000    
p <- 50

X <- cbind(matrix(rnorm(n * p), n, p))
true_beta <- c(rnorm(p))  
prob <- 1 / (1 + exp(-X %*% true_beta))
y <- rbinom(n, 1, prob)

beta_adam <- adaglm(X,y,fam_link = "binomial_logit", optimizer = "ADAM")
mse_adam <- mse(beta_adam$coefficients, true_beta)
mse_adam

beta_adam <- adaglm(X,y,fam_link = "binomial_logit", optimizer = "AdaGrad", learning_rate = 0.1)

mse_adam <- mse(beta_adam$coefficients, true_beta)
mse_adam

beta_adam <- adaglm(X,y,fam_link = "binomial_logit", optimizer = "AdaDelta")

mse_adam <- mse(beta_adam$coefficients, true_beta)
mse_adam

beta_adam <- adaglm(X,y,fam_link = "binomial_logit", optimizer = "AdaSmooth")

mse_adam <- mse(beta_adam$coefficients, true_beta)
mse_adam

X <- matrix(rnorm(n * p), nrow = n, ncol = p)
true_beta <- c(runif(5, -0.5, 0.5), rep(0, p - 5))
eta <- X %*% true_beta
mu <- exp(eta)
y <- rpois(n, lambda = mu)

beta_adam <- adaglm(X,y,fam_link = "poisson_log", optimizer = "ADAM")
mse_adam <- mse(beta_adam$coefficients, true_beta)
mse_adam

beta_adam <- adaglm(X,y,fam_link = "poisson_log", optimizer = "AdaGrad", learning_rate = 0.1)

mse_adam <- mse(beta_adam$coefficients, true_beta)
mse_adam

beta_adam <- adaglm(X,y,fam_link = "poisson_log", optimizer = "AdaDelta")

mse_adam <- mse(beta_adam$coefficients, true_beta)
mse_adam

beta_adam <- adaglm(X,y,fam_link = "poisson_log", optimizer = "AdaSmooth")

mse_adam <- mse(beta_adam$coefficients, true_beta)
mse_adam


X <- matrix(rnorm(n * p), nrow = n, ncol = p)
true_beta <- c(runif(5, -2, 2), rep(0, p - 5))
eta <- X %*% true_beta
sigma <- 1.0
y <- eta + rnorm(n, mean = 0, sd = sigma)

beta_adam <- adaglm(X,y,fam_link = "gaussian_identity", optimizer = "ADAM")
mse_adam <- mse(beta_adam$coefficients, true_beta)
mse_adam

beta_adam <- adaglm(X,y,fam_link = "gaussian_identity", optimizer = "AdaGrad", learning_rate = 0.1)

mse_adam <- mse(beta_adam$coefficients, true_beta)
mse_adam

beta_adam <- adaglm(X,y,fam_link = "gaussian_identity", optimizer = "AdaDelta")

mse_adam <- mse(beta_adam$coefficients, true_beta)
mse_adam

beta_adam <- adaglm(X,y,fam_link = "gaussian_identity", optimizer = "AdaSmooth")

mse_adam <- mse(beta_adam$coefficients, true_beta)
mse_adam


X <- matrix(rnorm(n * p), nrow = n, ncol = p)
true_beta <- c(runif(5, -1, 1), rep(0, p - 5))
eta <- X %*% true_beta
eta <- pmax(eta, 1e-3)
mu <- 1 / eta
lambda <- 2
y <- rgamma(n, shape = lambda, scale = mu / lambda)

beta_adam <- adaglm(X,y,fam_link = "Gamma_inverse", optimizer = "ADAM")
mse_adam <- mse(beta_adam$coefficients, true_beta)
mse_adam

beta_adam <- adaglm(X,y,fam_link = "Gamma_inverse", optimizer = "AdaGrad", stepsize = 0.1)

mse_adam <- mse(beta_adam$coefficients, true_beta)
mse_adam

beta_adam <- adaglm(X,y,fam_link = "Gamma_inverse", optimizer = "AdaDelta", stepsize = 0.1)

mse_adam <- mse(beta_adam$coefficients, true_beta)
mse_adam

beta_adam <- adaglm(X,y,fam_link = "Gamma_inverse", optimizer = "AdaSmooth", stepsize = 0.0001)
mse_adam <- mse(beta_adam$coefficients, true_beta)
mse_adam

