library(AdaGLM)
library(microbenchmark)
library(dplyr)

data(mtcars)
X = mtcars %>% mutate(intercept = rep(1,nrow(mtcars))) %>% 
  dplyr::select(intercept, wt, hp) %>%
  as.matrix
y = mtcars$mpg

beta_glm = summary(glm(y~X[,2:3], family = gaussian()))$coef[,1]

family = "gaussian_identity"
bench <- suppressWarnings(microbenchmark(
  beta_adam <- adaglm(X,y,fam_link = family, optimizer = "ADAM"),
  beta_adagrad <- adaglm(X,y,fam_link = family, optimizer = "AdaGrad"),
  beta_adadelta <- adaglm(X,y,fam_link = family, optimizer = "AdaDelta"),
  beta_adasmooth <- adaglm(X,y,fam_link = family, optimizer = "AdaSmooth"),
  beta_glm = summary(glm(y~X[,2:3], family = gaussian()))$coef[,1],
  times = 1L
))
exec_time_mtcars = summary(bench)$median
names(exec_time_mtcars) = c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")
exec_time_mtcars

beta_mat <- cbind(beta_adam, beta_adagrad, beta_adadelta, beta_adasmooth, as.numeric(beta_glm))
colnames(beta_mat) = c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")
beta_mat

loglik_mtcars = c(LogLik(X,y,fam_link = family, beta = beta_adam),
                 LogLik(X,y,fam_link = family, beta = beta_adagrad),
                 LogLik(X,y,fam_link = family, beta = beta_adadelta),
                 LogLik(X,y,fam_link = family, beta = beta_adasmooth),
                 LogLik(X,y,fam_link = family, beta = beta_glm))
names(loglik_mtcars) = c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")
loglik_mtcars

Deviance_mtcars = c(Deviance(X,y,fam_link = family, beta = beta_adam),
                   Deviance(X,y,fam_link = family, beta = beta_adagrad),
                   Deviance(X,y,fam_link = family, beta = beta_adadelta),
                   Deviance(X,y,fam_link = family, beta = beta_adasmooth),
                   Deviance(X,y,fam_link = family, beta = beta_glm))
names(Deviance_mtcars) = c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")
Deviance_mtcars
