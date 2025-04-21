# AdaGLM

This is a repository for the final project of BIOSTAT 815. We implement four adaptive learning rate methods -- AdaGrad, AdaDelta, ADAM, and AdaSmooth -- within a gradient descent framework to solve the MLE problem for four commonly used family-link pairs in GLMs.

## Installation

The latest version of this package can be installed using

```r
# install.packages("devtools") # if not already installed
devtools::install_github("lyvqian/AdaGLM")
```

## Function

The main function of this package is `adaglm()`

```r
adaglm(X, y, fam_link = "binomial_logit", optimizer = "ADAM", alpha = 0.01, rho = 0.99, max_iter = 1000, tol = 1e-6)
```

`X`: A matrix of predictors

`y`: A vector of responses

`fam_link`: One of "binomial_logit", "gaussian_identity", "Gamma_log", "poisson_log", meaning a distribution family and its corresponding link function

`optimizer`: One of "ADAM", "AdaGrad", "AdaSmooth", "AdaDelta"

`alpha`: Initial learning rate used in ADAM, AdaGrad, AdaSmooth

`rho`: Decay rate used in AdaDelta

`max_iter`: Maximum number of iterations

`tol`: Tolerence (defining convergence criteria)

For more information and examples, see the help page:

```r
?adaglm
```

