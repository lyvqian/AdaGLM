#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

arma::vec adagrad_logistic(const arma::mat& X, const arma::vec& y, double eta = 0.1, 
                           int max_iter = 1000, double epsilon = 1e-6) {
  int n = X.n_rows;
  int p = X.n_cols;
  arma::vec theta = arma::zeros(p); 
  arma::vec g_squared = arma::zeros(p); 
  
  for (int iter = 0; iter < max_iter; iter++) {
    arma::vec preds = 1.0 / (1.0 + arma::exp(-X * theta)); 
    arma::vec gradient = X.t() * (preds - y) / n; 
    
    g_squared += arma::square(gradient);
    arma::vec adjusted_alpha = eta / (arma::sqrt(g_squared) + epsilon);
    
    arma::vec update = adjusted_alpha % gradient;
    theta -= update;
    
    if (arma::norm(update, 2) < epsilon) break; 
  }
  return theta;
}

