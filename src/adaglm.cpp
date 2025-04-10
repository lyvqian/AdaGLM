#include <RcppArmadillo.h>
#include "families.h"
#include "optimizers.h"

using namespace Rcpp;


Family* get_family(std::string fam_link) {
  if (fam_link == "binomial_logit") return new BinomialLogit();
  if (fam_link == "gaussian_identity") return new GaussianIdentity();
  //if (fam_link == "Gamma_inverse") return new GammaInverse();
  if (fam_link == "Gamma_log") return new GammaLog();
  if (fam_link == "poisson_log") return new PoissonLog();
  stop("Unsupported family/link combination: " + fam_link);
}

//' Fit generalized linear regression models with adaptive learning rate algorithm
//'
//' @name adaglm
//'
//' @description adaglm( ) fits generalized linear regression models with a self-defined adaptive learning rate algorithm and stepsize. Default is ADAM with alpha=0.01.   
//'
//' @usage adaglm(X, y, fam_link = "binomial_logit", optimizer = "ADAM", 
//' alpha = 0.01, rho = 0.99, max_iter = 1000, tol = 1e-6)
//'
//' @param X  a matrix of predictors
//' @param y  a vector of responses
//' @param fam_link can be one of "binomial_logit", "gaussian_identity", "Gamma_log", "poisson_log", meaning a distribution family and its corresponding link function
//' @param optimizer can be one of "ADAM", "AdaGrad", "AdaSmooth", "AdaDelta"
//' @param alpha stepsize used in ADAM, AdaGrad, AdaSmooth
//' @param rho decay rate used in AdaDelta
//' @param max_iter maximum number of iterations
//' @param tol tolerence
//'
//' @section Note:
//' The number of rows of X must match the length of y.
//' If intercept is needed, a column of 1's has to be manually added to X.
//'
//' @return A vector of estimated regression coefficients 
//'
//' //@section See Also:
//' //
//'
//' @examples
//' ## Use the R built-in "quine" dataset
//' if (requireNamespace("dplyr", quietly = TRUE) && requireNamespace("microbenchmark", quietly = TRUE)) {
//'   data(quine, package = "MASS")
//'   X <- dplyr::mutate(quine,
//'     Ethnic = dplyr::case_when(Eth == "A" ~ 1, Eth == "N" ~ 0),
//'     Sex = dplyr::case_when(Sex == "F" ~ 1, Sex == "M" ~ 0),
//'     Age_F1 = dplyr::case_when(Age == "F1" ~ 1, TRUE ~ 0),
//'     Age_F2 = dplyr::case_when(Age == "F2" ~ 1, TRUE ~ 0),
//'     Age_F3 = dplyr::case_when(Age == "F3" ~ 1, TRUE ~ 0),
//'     Learner = dplyr::case_when(Lrn == "SL" ~ 1, Lrn == "AL" ~ 0),
//'     intercept = 1
//'   ) |> dplyr::select(intercept, Ethnic, Sex, Age_F1, Age_F2, Age_F3, Learner) |> as.matrix()
//'   y <- quine$Days
//'   family <- "poisson_log"
//'   bench <- suppressWarnings(microbenchmark::microbenchmark(
//'     beta_adam <- adaglm(X, y, fam_link = family, optimizer = "ADAM"),
//'     beta_adagrad <- adaglm(X, y, fam_link = family, optimizer = "AdaGrad"),
//'     beta_adadelta <- adaglm(X, y, fam_link = family, optimizer = "AdaDelta"),
//'     beta_adasmooth <- adaglm(X, y, fam_link = family, optimizer = "AdaSmooth"),
//'     beta_glm <- summary(glm(y ~ X[, 2:7], family = poisson()))$coef[, 1],
//'     times = 1L
//'   ))
//' }
//' 
//' exec_time_quine = summary(bench)$median
//' names(exec_time_quine) = c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")
//' exec_time_quine
//' 
//' beta_mat <- cbind(beta_adam, 
//' beta_adagrad, beta_adadelta, 
//' beta_adasmooth, as.numeric(beta_glm))
//' colnames(beta_mat) = c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")
//' beta_mat
//' 
//' loglik_quine = c(LogLik(X,y,fam_link = family, beta = beta_adam),
//'                 LogLik(X,y,fam_link = family, beta = beta_adagrad),
//'                 LogLik(X,y,fam_link = family, beta = beta_adadelta),
//'                 LogLik(X,y,fam_link = family, beta = beta_adasmooth),
//'                 LogLik(X,y,fam_link = family, beta = beta_glm))
//'                 names(loglik_quine) = c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")
//' loglik_quine
//' 
//' Deviance_quine = c(Deviance(X,y,fam_link = family, beta = beta_adam),
//'                  Deviance(X,y,fam_link = family, beta = beta_adagrad),
//'                  Deviance(X,y,fam_link = family, beta = beta_adadelta),
//'                  Deviance(X,y,fam_link = family, beta = beta_adasmooth),
//'                  Deviance(X,y,fam_link = family, beta = beta_glm))
//'                  names(Deviance_quine) = c("ADAM", "AdaGrad", "AdaDelta", "AdaSmooth", "glm_fn")
//' Deviance_quine
//' 
//' @export
//'

// [[Rcpp::export]]
arma::vec adaglm(const arma::mat& X, const arma::vec& y,
            std::string fam_link="binomial_logit", std::string optimizer="ADAM",
            double alpha = 0.01, double rho = 0.99, int max_iter = 1000, double tol = 1e-6) {
  
  Family* family = get_family(fam_link);
  
  OptimizerConfig cfg;
  cfg.alpha = alpha;
  cfg.rho = rho; 
  cfg.max_iter = max_iter;
  cfg.epsilon = tol;
  
  if (optimizer == "ADAM") {
    cfg.method = ADAM;
  } else if (optimizer == "AdaGrad") {
    cfg.method = AdaGrad;
  } else if (optimizer == "AdaSmooth") {
    cfg.method = AdaSmooth;
  } else if (optimizer == "AdaDelta") {
    cfg.method = AdaDelta;
  } else {
    Rcpp::stop("Unknown optimizer: " + optimizer);
  }
  
  arma::vec beta = run_optimizer(X, y, family, cfg);
  delete family;
  
  return beta;
}


