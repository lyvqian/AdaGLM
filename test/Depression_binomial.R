library(AdaGLM)
library(microbenchmark)
library(dplyr)

depression = read.csv("student_depression_dataset.csv")

X = depression %>% mutate(pressure = Work.Pressure+Academic.Pressure,
                          satisfaction = Study.Satisfaction+Job.Satisfaction,
                          gender = case_when(Gender == "Male" ~ 0, Gender == "Female" ~1),
                          suicide = case_when(Have.you.ever.had.suicidal.thoughts.. == "Yes" ~ 1, Have.you.ever.had.suicidal.thoughts.. == "No" ~0),
                          familyhistory = case_when(Family.History.of.Mental.Illness == "Yes" ~ 1, Family.History.of.Mental.Illness == "No" ~0),
                          intercept = rep(1,nrow(depression))) %>% 
  select(intercept, gender, Age, pressure, satisfaction, suicide, familyhistory) %>%
  as.matrix
y = depression$Depression

beta_glm = summary(glm(y~X[,2:7], family = binomial()))$coef[,1]

family = "binomial_logit"
bench <- suppressWarnings(microbenchmark(
  beta_adam <- adaglm(X,y,fam_link = family, optimizer = "ADAM"),
  beta_adagrad <- adaglm(X,y,fam_link = family, optimizer = "AdaGrad"),
  beta_adadelta <- adaglm(X,y,fam_link = family, optimizer = "AdaDelta"),
  beta_adasmooth <- adaglm(X,y,fam_link = family, optimizer = "AdaSmooth"),
  beta_glm = summary(glm(y~X[,2:7], family = binomial()))$coef[,1],
  times = 1L
))

exec_time_Depression = summary(bench)$median
exec_time_Depression

beta_mat <- cbind(beta_adam$coefficients, beta_adagrad$coefficients, beta_adadelta$coefficients, beta_adasmooth$coefficients, as.numeric(beta_glm))
colnames(beta_mat) = c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")
beta_mat

loglik_depression = c(LogLik(X,y,fam_link = family, beta = beta_adam$coefficients),
                 LogLik(X,y,fam_link = family, beta = beta_adagrad$coefficients),
                 LogLik(X,y,fam_link = family, beta = beta_adadelta$coefficients),
                 LogLik(X,y,fam_link = family, beta = beta_adasmooth$coefficients),
                 LogLik(X,y,fam_link = family, beta = beta_glm))
names(loglik_depression) = c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")
loglik_depression

Deviance_depression = c(Deviance(X,y,fam_link = family, beta = beta_adam$coefficients),
                   Deviance(X,y,fam_link = family, beta = beta_adagrad$coefficients),
                   Deviance(X,y,fam_link = family, beta = beta_adadelta$coefficients),
                   Deviance(X,y,fam_link = family, beta = beta_adasmooth$coefficients),
                   Deviance(X,y,fam_link = family, beta = beta_glm))
names(Deviance_depression) = c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")
Deviance_depression

