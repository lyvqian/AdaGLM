library(AdaptiveLearningRate)
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

mse <- function(beta_est, beta_true) {
  mean((beta_est - beta_true)^2)
}

beta_glm = summary(glm(y~X[,2:7], family = binomial()))$coef[,1]

beta_adam <- adaglm(X,y,fam_link = "binomial_logit", optimizer = "ADAM")
mse_adam <- mse(beta_adam$coefficients, beta_glm)
mse_adam

beta_adagrad <- adaglm(X,y,fam_link = "binomial_logit", optimizer = "AdaGrad")
mse_adagrad <- mse(beta_adagrad$coefficients, beta_glm)
mse_adagrad

beta_adadelta <- adaglm(X,y,fam_link = "binomial_logit", optimizer = "AdaDelta")
mse_adadelta <- mse(beta_adadelta$coefficients, beta_glm)
mse_adadelta

beta_adasmooth <- adaglm(X,y,fam_link = "binomial_logit", optimizer = "AdaSmooth")
mse_adasmooth <- mse(beta_adasmooth$coefficients, beta_glm)
mse_adasmooth