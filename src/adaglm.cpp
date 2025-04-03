#include <RcppArmadillo.h>
#include "families.h"
#include "optimizers.h"

using namespace Rcpp;


Family* get_family(std::string fam_link) {
  if (fam_link == "binomial_logit") return new BinomialLogit();
  if (fam_link == "gaussian_identity") return new GaussianIdentity();
  if (fam_link == "Gamma_inverse") return new GammaInverse();
  if (fam_link == "poisson_log") return new PoissonLog();
  stop("Unsupported family/link combination: " + fam_link);
}

// [[Rcpp::export]]
List adaglm(const arma::mat& X, const arma::vec& y,
                 std::string fam_link, std::string optimizer,
                 double stepsize = 0.01, int max_iter = 1000, double tol = 1e-6) {
  
  Family* family = get_family(fam_link);
  
  OptimizerConfig cfg;
  cfg.alpha = stepsize;
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
  
  return List::create(Named("coefficients") = beta);
}


