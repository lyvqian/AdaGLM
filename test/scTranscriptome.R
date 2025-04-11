library(AdaGLM)
library(microbenchmark)
library(dplyr)

tumor <- read.csv("GSE81861_CRC_tumor_all_cells_FPKM.csv.gz")
NM <- read.csv("GSE81861_CRC_NM_all_cells_FPKM.csv.gz")
data = as.data.frame(log2(t(cbind(tumor[,-1], NM[,-1]))+1))
gene = tumor[,1]
y = c(rep(1,ncol(tumor)-1),rep(0,ncol(NM)-1))
variances <- apply(data, 2, var)
colnames(data) = gene
X = as.matrix(data[,-which(variances < 15)]) # select 266 genes * 590 samples

family = "binomial_logit"
bench <- suppressWarnings(microbenchmark(
  beta_adam <- adaglm(X,y,fam_link = family, optimizer = "ADAM"),
  beta_adagrad <- adaglm(X,y,fam_link = family, optimizer = "AdaGrad"),
  beta_adadelta <- adaglm(X,y,fam_link = family, optimizer = "AdaDelta"),
  beta_adasmooth <- adaglm(X,y,fam_link = family, optimizer = "AdaSmooth"),
  beta_glm = glm(y~X-1, family = binomial())$coef,
  times = 1L
))

exec_time_Depression = summary(bench)$median
exec_time_Depression

beta_mat <- cbind(beta_adam, beta_adagrad, beta_adadelta, beta_adasmooth, beta_glm)
colnames(beta_mat) = c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")
beta_mat

loglik_depression = c(LogLik(X,y,fam_link = family, beta = beta_adam),
                      LogLik(X,y,fam_link = family, beta = beta_adagrad),
                      LogLik(X,y,fam_link = family, beta = beta_adadelta),
                      LogLik(X,y,fam_link = family, beta = beta_adasmooth),
                      LogLik(X,y,fam_link = family, beta = beta_glm))
names(loglik_depression) = c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")
loglik_depression

Deviance_depression = c(Deviance(X,y,fam_link = family, beta = beta_adam),
                        Deviance(X,y,fam_link = family, beta = beta_adagrad),
                        Deviance(X,y,fam_link = family, beta = beta_adadelta),
                        Deviance(X,y,fam_link = family, beta = beta_adasmooth),
                        Deviance(X,y,fam_link = family, beta = beta_glm))
names(Deviance_depression) = c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")
Deviance_depression

