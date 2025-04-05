library(AdaptiveLearningRate)
library(dplyr)

income = read.csv("census_income.csv")

X <- model.matrix(
  ~ marital_group + log_age + education_level + race_two + sex + occupation_group + native_us,
  data = income
)
y <- income$income_two

model <- glm.fit(x = X, y = y, family = binomial())
beta_glm <- coef(model) # does not converge, coefficients almost 0 except the intercept 

X_inter <- cbind(rep(1,nrow(X)), X)
family = "binomial_logit"

beta_adam <- adaglm(X_inter,y,fam_link = family, optimizer = "ADAM")
beta_adagrad <- adaglm(X_inter,y,fam_link = family, optimizer = "AdaGrad")
beta_adadelta <- adaglm(X_inter,y,fam_link = family, optimizer = "AdaDelta")
beta_adasmooth <- adaglm(X_inter,y,fam_link = family, optimizer = "AdaSmooth")

beta_mat <- cbind(beta_adam$coefficients, beta_adagrad$coefficients, beta_adadelta$coefficients, beta_adasmooth$coefficients, as.numeric(beta_glm))
colnames(beta_mat) = c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")
beta_mat

