library(AdaGLM)
library(microbenchmark)
library(dplyr)
library(ggplot2)
library(sgd)

# Read files and combine the 2 files into 1
tumor <- read.csv("GSE81861_CRC_tumor_all_cells_FPKM.csv.gz")
NM <- read.csv("GSE81861_CRC_NM_all_cells_FPKM.csv.gz")
data = as.data.frame(log2(t(cbind(tumor[,-1], NM[,-1]))+1))
gene = tumor[,1]
y = c(rep(1,ncol(tumor)-1),rep(0,ncol(NM)-1)) # Use whether is tumor as the outcome

# Perform PCA to reduce dimension
pca <- prcomp(data)
var_explained <- pca$sdev^2
prop_var <- var_explained / sum(var_explained)
cum_var <- cumsum(prop_var)
df <- data.frame(
  PC = 1:length(cum_var),
  CumulativeVariance = cum_var
)
# Make a plot of the cumulative variance of PCA
ggplot(df, aes(x = PC, y = CumulativeVariance)) +
  geom_line(color = "#00274C", linewidth = 1) +
  geom_hline(yintercept = 0.3, linetype = "dashed", color = "red") +
  labs(
    title = "Cumulative Variance Explained by PCA",
    x = "Number of Principal Components",
    y = "Cumulative Proportion of Variance Explained"
  ) +
  theme_minimal()
X = pca$x[,1:48] # Select 48 PCs to explain >30% cumulative variance

# Compute glm for adaglm(), glm() and sgd()
family = "binomial_logit"
bench <- suppressWarnings(microbenchmark(
  beta_adagrad <- adaglm(X,y,fam_link = family, optimizer = "AdaGrad")$coef,
  beta_adadelta <- adaglm(X,y,fam_link = family, optimizer = "AdaDelta")$coef,
  beta_adam <- adaglm(X,y,fam_link = family, optimizer = "ADAM")$coef,
  beta_adasmooth <- adaglm(X,y,fam_link = family, optimizer = "AdaSmooth")$coef,
  beta_glm <- glm(y~X-1, family = binomial())$coef,
  beta_sgd <- sgd(y~X-1, model = "glm", model.control = list(family = binomial()),
                  sgd.control = list(method = "sgd", lr = "adagrad", lr.control = c(0.01, 1e-6)))$coef,
  times = 1L
))

# Compare the execution time
exec_time = summary(bench)$median
names(exec_time) = c("AdaGrad", "AdaDelta", "ADAM", "AdaSmooth", "glm_fn", "sgd")
exec_time

# Calculate the accuracy
beta_mat <- cbind(beta_adagrad, beta_adadelta, beta_adam, beta_adasmooth, beta_glm, beta_sgd)
colnames(beta_mat) = c("AdaGrad", "AdaDelta", "ADAM", "AdaSmooth", "glm_fn", "sgd")

acc_adadelta = 1-sum(abs(ifelse(1/(1+exp(-X %*% beta_adadelta)) > 0.5, 1, 0) - y))/length(y)
acc_adam = 1-sum(abs(ifelse(1/(1+exp(-X %*% beta_adam)) > 0.5, 1, 0) - y))/length(y)
acc_adagrad = 1-sum(abs(ifelse(1/(1+exp(-X %*% beta_adagrad)) > 0.5, 1, 0) - y))/length(y)
acc_adasmooth = 1-sum(abs(ifelse(1/(1+exp(-X %*% beta_adasmooth)) > 0.5, 1, 0) - y))/length(y)
acc_glm = 1-sum(abs(ifelse(1/(1+exp(-X %*% beta_glm)) > 0.5, 1, 0) - y))/length(y)
acc_sgd = 1-sum(abs(ifelse(1/(1+exp(-X %*% beta_sgd)) > 0.5, 1, 0) - y))/length(y)

acc = c(acc_adagrad, acc_adadelta, acc_adam, acc_adasmooth, acc_glm, acc_sgd)*100
names(acc) = c("AdaGrad", "AdaDelta", "ADAM", "AdaSmooth", "glm_fn", "sgd")
acc

# Calculate the log-likelihood
loglik = c(
  LogLik(X, y, fam_link = family, beta = beta_adagrad),
  LogLik(X, y, fam_link = family, beta = beta_adadelta),
  LogLik(X, y, fam_link = family, beta = beta_adam),
  LogLik(X, y, fam_link = family, beta = beta_adasmooth),
  LogLik(X, y, fam_link = family, beta = beta_glm),
  LogLik(X, y, fam_link = family, beta = beta_sgd)
)
names(loglik) = c("AdaGrad", "AdaDelta", "ADAM", "AdaSmooth", "glm_fn", "sgd")
loglik

# mutiply the beta of adam by rotation matrix and select the top poisitive and top negative beta of features
V <- pca$rotation[, 1:48] 
gene_contrib <- V %*% beta_adam
gene_name = sub("^[^_]*_([^_]*)_.*$", "\\1", gene)
row.names(gene_contrib) = gene_name
top_pos <- sort(gene_contrib[,1], decreasing = TRUE)[1:10]  # most positively associated
top_neg <- sort(gene_contrib[,1], decreasing = FALSE)[1:10] # most negatively associated
top_pos
top_neg

# plot top positive betas
df = data.frame(name = names(top_pos), value = as.numeric(top_pos))
df$name <- factor(df$name, levels = df$name[order(df$value)])
ggplot(df, 
       aes(x = value, y = name)) +
  geom_col(fill = "#00274C") +
  labs(x = "Beta", y = "Features", title = "Top Positive Beta") +
  theme_bw()
ggsave(file = "top_pos.png", width = 4, height = 4)

# plot top negative betas
df = data.frame(name = names(top_neg), value = as.numeric(top_neg))
df$name <- factor(df$name, levels = df$name[order(df$value, decreasing = T)])
ggplot(df, 
       aes(x = value, y = name)) +
  geom_col(fill = "#00274C") +
  labs(x = "Beta", y = "Features", title = "Top Negative Beta") +
  theme_bw()
ggsave(file = "top_neg.png", width = 4, height = 4)
