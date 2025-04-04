library(AdaptiveLearningRate)
library(dplyr)
library(microbenchmark)
library(MASS)

data(quine)
X = quine %>% mutate(Ethnic = case_when(Eth == "A" ~ 1, Eth == "N" ~ 0),
                     Sex = case_when(Sex == "F" ~ 1, Sex == "M" ~ 0),
                     Age_F1 = case_when(Age == "F1" ~ 1, .default = 0),
                     Age_F2 = case_when(Age == "F2" ~ 1, .default = 0),
                     Age_F3 = case_when(Age == "F3" ~ 1, .default = 0),
                     Learner = case_when(Lrn == "SL" ~ 1, Lrn == "AL" ~ 0),
                     intercept = rep(1,nrow(quine))) %>% 
  dplyr::select(intercept, Ethnic, Sex, Age_F1, Age_F2, Age_F3, Learner) %>%
  as.matrix
y = quine$Days


family = "poisson_log"
bench <- suppressWarnings(microbenchmark(
  beta_adam <- adaglm(X,y,fam_link = family, optimizer = "ADAM"),
  beta_adagrad <- adaglm(X,y,fam_link = family, optimizer = "AdaGrad"),
  beta_adadelta <- adaglm(X,y,fam_link = family, optimizer = "AdaDelta"),
  beta_adasmooth <- adaglm(X,y,fam_link = family, optimizer = "AdaSmooth"),
  beta_glm = summary(glm(y~X[,2:7], family = poisson()))$coef[,1],
  times = 1L
))

exec_time_quine = summary(bench)$median
names(loglik_quine) = c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")

loglik_quine = c(LogLik(X,y,fam_link = family, beta = beta_adam$coefficients),
                 LogLik(X,y,fam_link = family, beta = beta_adagrad$coefficients),
                 LogLik(X,y,fam_link = family, beta = beta_adadelta$coefficients),
                 LogLik(X,y,fam_link = family, beta = beta_adasmooth$coefficients),
                 LogLik(X,y,fam_link = family, beta = beta_glm))
names(loglik_quine) = c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")

Deviance_quine = c(Deviance(X,y,fam_link = family, beta = beta_adam$coefficients),
                   Deviance(X,y,fam_link = family, beta = beta_adagrad$coefficients),
                   Deviance(X,y,fam_link = family, beta = beta_adadelta$coefficients),
                   Deviance(X,y,fam_link = family, beta = beta_adasmooth$coefficients),
                   Deviance(X,y,fam_link = family, beta = beta_glm))
names(Deviance_quine) = c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")
