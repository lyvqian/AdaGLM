#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
double LogLik(const arma::mat& X, const arma::vec& y,
              std::string fam_link, const arma::vec& beta) {
  arma::vec eta = X * beta;
  arma::vec mu;
  double loglik = 0.0;
  
  if (fam_link == "poisson_log") {
    mu = arma::exp(eta);
    loglik = arma::sum(y % arma::log(mu) - mu - lgamma(y + 1));
  }
  else if (fam_link == "gaussian_identity") {
    mu = eta;
    arma::vec resid = y - mu;
    loglik = -0.5 * arma::sum(arma::square(resid));
  }
  else if (fam_link == "binomial_logit") {
    mu = 1 / (1 + arma::exp(-eta));
    loglik = arma::sum(y % arma::log(mu) + (1 - y) % arma::log(1 - mu));
  }
  else if (fam_link == "Gamma_inverse") {
    mu = arma::exp(eta);
    // Shape parameter assumed to be known or absorbed
    // This assumes a canonical link (log)
    loglik = arma::sum(-y / mu - arma::log(mu));  // Simplified
  }
  else {
    stop("Unsupported family/link combination: " + fam_link);
  }
  
  return loglik;
}

// [[Rcpp::export]]
double Deviance(const arma::mat& X,
                    const arma::vec& y,
                    const arma::vec& beta,
                    std::string fam_link) {
  
  arma::vec eta = X * beta;
  arma::vec mu;
  double dev = 0.0;
  
  if (fam_link == "poisson_log") {
    mu = arma::exp(eta);
    for (size_t i = 0; i < y.n_elem; ++i) {
      if (y[i] == 0) {
        dev += 2.0 * mu[i];
      } else {
        dev += 2.0 * (y[i] * std::log(y[i] / mu[i]) - (y[i] - mu[i]));
      }
    }
  }
  else if (fam_link == "gaussian_identity") {
    mu = eta;
    arma::vec resid = y - mu;
    dev = arma::sum(arma::square(resid));  // Residual sum of squares
  }
  else if (fam_link == "binomial_logit") {
    mu = 1 / (1 + arma::exp(-eta));
    for (size_t i = 0; i < y.n_elem; ++i) {
      if (y[i] == 0) {
        dev += 2.0 * std::log(1.0 / (1.0 - mu[i]));
      } else if (y[i] == 1) {
        dev += 2.0 * std::log(1.0 / mu[i]);
      } else {
        dev += 2.0 * (y[i] * std::log(y[i] / mu[i]) +
          (1 - y[i]) * std::log((1 - y[i]) / (1 - mu[i])));
      }
    }
  }
  else if (fam_link == "Gamma_inverse") {
    mu = arma::exp(eta);
    for (size_t i = 0; i < y.n_elem; ++i) {
      dev += 2.0 * ((y[i] - mu[i]) / mu[i] - std::log(y[i] / mu[i]));
    }
  }
  else {
    stop("Unsupported family/link combination: " + fam_link);
  }
  return dev;
}

