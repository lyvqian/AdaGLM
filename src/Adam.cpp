#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

arma::vec adam_logistic(const arma::mat& X, const arma::vec& y, 
                        double alpha = 0.01, int max_iter = 1000, 
                        double beta1 = 0.9, double beta2 = 0.999, // default values are set according to the paper
                        double epsilon = 1e-8) {
  int n = X.n_rows;
  int p = X.n_cols;
  arma::vec theta = arma::zeros(p); 
  arma::vec m = arma::zeros(p); 
  arma::vec v = arma::zeros(p); 
  
  for (int iter = 1; iter <= max_iter; iter++) {
    arma::vec pred = 1.0 / (1.0 + arma::exp(-X * theta)); 
    arma::vec gradient = X.t() * (pred - y) / n; 
    
    m = beta1 * m + (1 - beta1) * gradient;
    v = beta2 * v + (1 - beta2) * arma::square(gradient);

    arma::vec m_hat = m / (1 - std::pow(beta1, iter));
    arma::vec v_hat = v / (1 - std::pow(beta2, iter));

    theta -= alpha * m_hat / (arma::sqrt(v_hat) + epsilon);

    if (arma::norm(gradient, 2) < epsilon) break;
  }
  
  return theta;
}

