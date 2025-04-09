library(AdaGLM)
library(dplyr)

income = read.csv("census_income.csv")

X <- model.matrix(
  ~ age + sex + education_level,
  data = income
)
y <- income$income_two

model <- glm(income_two ~ age + sex + education_level, data = income[1:500, ], family = binomial())
beta_glm <- coef(model) # does not converge, coefficients almost 0 except the intercept 

#X_inter <- cbind(rep(1,nrow(X)), X[,2])
#X_inter <- cbind(rep(1,500), X[1:500,2])
X_inter <- X[1:500,]
family = "binomial_logit"

beta_adam <- adaglm(X_inter,y[1:500],fam_link = family, optimizer = "ADAM",alpha=0.1)
beta_adagrad <- adaglm(X_inter,y[1:500],fam_link = family, optimizer = "AdaGrad", alpha=0.1)
beta_adadelta <- adaglm(X_inter,y[1:500],fam_link = family, optimizer = "AdaDelta", rho=0.99)
beta_adasmooth <- adaglm(X_inter,y[1:500],fam_link = family, optimizer = "AdaSmooth", alpha=0.1)

beta_mat <- cbind(beta_adam$coefficients, beta_adagrad$coefficients, beta_adadelta$coefficients, beta_adasmooth$coefficients, as.numeric(beta_glm))
colnames(beta_mat) = c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")
beta_mat

