# AdaGLM

This is a repository for the final project of BIOSTAT 815. We implement four adaptive learning rate methods -- AdaGrad, AdaDelta, ADAM, and AdaSmooth -- within a gradient descent framework to solve the MLE problem for four commonly used family-link pairs in GLMs.

## Installation

The latest version of this package can be installed using

```r
# install.packages("devtools") # if not already installed
devtools::install_github("lyvqian/AdaGLM")
```

## Function

The main function of this package is `adaglm()`

```r
adaglm(X, y, fam_link = "binomial_logit", optimizer = "ADAM", alpha = 0.01, rho = 0.99, max_iter = 1000, tol = 1e-6)
```

`X`: A matrix of predictors

`y`: A vector of responses

`fam_link`: One of "binomial_logit", "gaussian_identity", "Gamma_log", "poisson_log", meaning a distribution family and its corresponding link function

`optimizer`: One of "ADAM", "AdaGrad", "AdaSmooth", "AdaDelta"

`alpha`: Initial learning rate used in ADAM, AdaGrad, AdaSmooth

`rho`: Decay rate used in AdaDelta

`max_iter`: Maximum number of iterations

`tol`: Tolerence (defining convergence criteria)

For more information and examples, see the help page:

```r
?adaglm
```

## Simulations

Folder [Simulation studies](https://github.com/lyvqian/AdaGLM/tree/main/test/Simulation%20Studies) contains all the codes for simulations, pre-generated simulation data, and boxplots used in our report. Packages needed in this code are:
- Matrix
- tidyr
- microbenchmark
- dplyr
- ggplot2

## Real Data Analysis

The datasets can be downloaded via the GEO website [GSE81861](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE81861). The 2 datasets needed in real data analysis are 'GSE81861_CRC_NM_all_cells_FPKM.csv.gz' and 'GSE81861_CRC_tumor_all_cells_FPKM.csv.gz'. We also provided the datasets on our GitHub: [Normal Mucosa](https://github.com/lyvqian/AdaGLM/blob/main/GSE81861_CRC_NM_all_cells_FPKM.csv.gz) and [Tumor](https://github.com/lyvqian/AdaGLM/blob/main/GSE81861_CRC_tumor_all_cells_FPKM.csv.gz).

The codes for real data analysis can be found in [scTranscriptome.R](https://github.com/lyvqian/AdaGLM/blob/main/test/scTranscriptome.R). Packages needed in this code are:
- microbenchmark
- dplyr
- ggplot2
- sgd

