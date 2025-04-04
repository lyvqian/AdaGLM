library(AdaptiveLearningRate)
library(microbenchmark)
library(dplyr)

data(airquality)
air <- na.omit(airquality)
hist(air$Ozone)
X = air %>% mutate(intercept = rep(1,nrow(air))) %>% 
  dplyr::select(intercept, Solar.R, Temp, Wind) %>%
  as.matrix
y = air$Ozone

beta_glm = summary(glm(y~X[,2:4], family = Gamma(link = "inverse")))$coef[,1]

family = "Gamma_inverse"
bench <- suppressWarnings(microbenchmark(
  beta_adam <- adaglm(X,y,fam_link = family, optimizer = "ADAM", alpha=0.0001),
  beta_adagrad <- adaglm(X,y,fam_link = family, optimizer = "AdaGrad", alpha=0.0001),
  beta_adadelta <- adaglm(X,y,fam_link = family, optimizer = "AdaDelta", rho=0.90),
  beta_adasmooth <- adaglm(X,y,fam_link = family, optimizer = "AdaSmooth", alpha=0.0001),
  beta_glm = summary(glm(y~X[,2:4], family = Gamma(link = "log")))$coef[,1],
  times = 1L
))

exec_time_insurance = summary(bench)$median
exec_time_insurance

beta_mat <- cbind(beta_adam$coefficients, beta_adagrad$coefficients, beta_adadelta$coefficients, beta_adasmooth$coefficients, as.numeric(beta_glm))
colnames(beta_mat) = c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")
beta_mat

loglik_insurance = c(LogLik(X,y,fam_link = family, beta = beta_adam$coefficients),
                 LogLik(X,y,fam_link = family, beta = beta_adagrad$coefficients),
                 LogLik(X,y,fam_link = family, beta = beta_adadelta$coefficients),
                 LogLik(X,y,fam_link = family, beta = beta_adasmooth$coefficients),
                 LogLik(X,y,fam_link = family, beta = beta_glm))
names(loglik_insurance) = c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")
loglik_insurance

Deviance_insurance = c(Deviance(X,y,fam_link = family, beta = beta_adam$coefficients),
                   Deviance(X,y,fam_link = family, beta = beta_adagrad$coefficients),
                   Deviance(X,y,fam_link = family, beta = beta_adadelta$coefficients),
                   Deviance(X,y,fam_link = family, beta = beta_adasmooth$coefficients),
                   Deviance(X,y,fam_link = family, beta = beta_glm))
names(Deviance_insurance) = c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")
Deviance_insurance


