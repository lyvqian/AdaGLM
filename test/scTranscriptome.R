library(AdaGLM)
library(microbenchmark)
library(dplyr)
library(xtable)
library(ggplot2)
library(sgd)

tumor <- read.csv("GSE81861_CRC_tumor_all_cells_FPKM.csv.gz")
NM <- read.csv("GSE81861_CRC_NM_all_cells_FPKM.csv.gz")
data = as.data.frame(log2(t(cbind(tumor[,-1], NM[,-1]))+1))
gene = tumor[,1]
y = c(rep(1,ncol(tumor)-1),rep(0,ncol(NM)-1))
pca <- prcomp(data)
var_explained <- pca$sdev^2
prop_var <- var_explained / sum(var_explained)
cum_var <- cumsum(prop_var)
df <- data.frame(
  PC = 1:length(cum_var),
  CumulativeVariance = cum_var
)
ggplot(df, aes(x = PC, y = CumulativeVariance)) +
  geom_line(color = "#00274C", linewidth = 1) +
  geom_hline(yintercept = 0.3, linetype = "dashed", color = "red") +
  labs(
    title = "Cumulative Variance Explained by PCA",
    x = "Number of Principal Components",
    y = "Cumulative Proportion of Variance Explained"
  ) +
  theme_minimal()
X = pca$x[,1:48]

family = "binomial_logit"
bench <- suppressWarnings(microbenchmark(
  beta_adagrad <- adaglm(X,y,fam_link = family, optimizer = "AdaGrad"),
  beta_adadelta <- adaglm(X,y,fam_link = family, optimizer = "AdaDelta"),
  beta_adam <- adaglm(X,y,fam_link = family, optimizer = "ADAM"),
  beta_adasmooth <- adaglm(X,y,fam_link = family, optimizer = "AdaSmooth"),
  beta_glm <- glm(y~X-1, family = binomial())$coef,
  beta_sgd <- sgd(y~X-1, model = "glm", model.control = list(family = binomial()),
                  sgd.control = list(method = "sgd", lr = "adagrad", lr.control = c(0.01, 1e-6)))$coef,
  times = 1L
))

exec_time = summary(bench)$median
names(exec_time) = c("AdaGrad", "AdaDelta", "ADAM", "AdaSmooth", "glm_fn", "sgd")

print(xtable(as.data.frame(t(exec_time))), include.rownames = FALSE)
exec_time

beta_mat <- cbind(beta_adagrad, beta_adadelta, beta_adam, beta_adasmooth, beta_glm, beta_sgd)
colnames(beta_mat) = c("AdaGrad", "AdaDelta", "ADAM", "AdaSmooth", "glm_fn", "sgd")
beta_mat

acc_adadelta = 1-sum(abs(ifelse(1/(1+exp(-X %*% beta_adadelta)) > 0.5, 1, 0) - y))/length(y)
acc_adam = 1-sum(abs(ifelse(1/(1+exp(-X %*% beta_adam)) > 0.5, 1, 0) - y))/length(y)
acc_adagrad = 1-sum(abs(ifelse(1/(1+exp(-X %*% beta_adagrad)) > 0.5, 1, 0) - y))/length(y)
acc_adasmooth = 1-sum(abs(ifelse(1/(1+exp(-X %*% beta_adasmooth)) > 0.5, 1, 0) - y))/length(y)
acc_glm = 1-sum(abs(ifelse(1/(1+exp(-X %*% beta_glm)) > 0.5, 1, 0) - y))/length(y)
acc_sgd = 1-sum(abs(ifelse(1/(1+exp(-X %*% beta_sgd)) > 0.5, 1, 0) - y))/length(y)

acc = c(acc_adagrad, acc_adadelta, acc_adam, acc_adasmooth, acc_glm, acc_sgd)*100
names(acc) = c("AdaGrad", "AdaDelta", "ADAM", "AdaSmooth", "glm_fn", "sgd")
print(xtable(as.data.frame(t(acc))), include.rownames = FALSE)

loglik = c(
  LogLik(X, y, fam_link = family, beta = beta_adagrad),
  LogLik(X, y, fam_link = family, beta = beta_adadelta),
  LogLik(X, y, fam_link = family, beta = beta_adam),
  LogLik(X, y, fam_link = family, beta = beta_adasmooth),
  LogLik(X, y, fam_link = family, beta = beta_glm),
  LogLik(X, y, fam_link = family, beta = beta_sgd)
)
names(loglik) = c("AdaGrad", "AdaDelta", "ADAM", "AdaSmooth", "glm_fn", "sgd")
print(xtable(as.data.frame(t(loglik))), include.rownames = FALSE)

# Deviance_depression = c(Deviance(X,y,fam_link = family, beta = beta_adam),
#                         Deviance(X,y,fam_link = family, beta = beta_adagrad),
#                         Deviance(X,y,fam_link = family, beta = beta_adadelta),
#                         Deviance(X,y,fam_link = family, beta = beta_adasmooth),
#                         Deviance(X,y,fam_link = family, beta = beta_glm))
# names(Deviance_depression) = c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")
# Deviance_depression

