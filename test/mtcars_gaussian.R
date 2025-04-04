library(AdaptiveLearningRate)
library(microbenchmark)
library(dplyr)

data(mtcars)
X = mtcars %>% mutate(intercept = rep(1,nrow(mtcars))) %>% 
  dplyr::select(intercept, wt, hp) %>%
  as.matrix
y = mtcars$mpg

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
names(loglik_mtcars) = c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")


loglik_mtcars = c(LogLik(X,y,fam_link = family, beta = beta_adam$coefficients),
                 LogLik(X,y,fam_link = family, beta = beta_adagrad$coefficients),
                 LogLik(X,y,fam_link = family, beta = beta_adadelta$coefficients),
                 LogLik(X,y,fam_link = family, beta = beta_adasmooth$coefficients),
                 LogLik(X,y,fam_link = family, beta = beta_glm))
names(loglik_mtcars) = c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")

Deviance_mtcars = c(Deviance(X,y,fam_link = family, beta = beta_adam$coefficients),
                   Deviance(X,y,fam_link = family, beta = beta_adagrad$coefficients),
                   Deviance(X,y,fam_link = family, beta = beta_adadelta$coefficients),
                   Deviance(X,y,fam_link = family, beta = beta_adasmooth$coefficients),
                   Deviance(X,y,fam_link = family, beta = beta_glm))
names(Deviance_mtcars) = c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")
