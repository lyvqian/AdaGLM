#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

arma::vec adadelta_logistic(const arma::mat& X, const arma::vec& y,
                             int max_iter = 1000, double epsilon = 1e-6,
                             double rho = 0.95) {
  int n = X.n_rows;
  int p = X.n_cols;
  arma::vec theta = arma::zeros(p);
  arma::vec E_g2 = zeros(p); 
  arma::vec E_delta_theta2 = zeros(p);
  arma::vec prev_theta = theta;
  
  arma::vec deltaX2 = zeros(p);
  
  arma::vec preds = zeros(p);
  arma::vec gradient = zeros(p);
  
  for (int iter = 0; iter < max_iter; iter++) {
    preds = 1.0 / (1.0 + arma::exp(-X * theta)); 
    gradient = X.t() * (preds - y) / n; 
    
    arma::vec gradient2 = gradient % gradient;
    E_g2 = rho * E_g2 + (1 - rho) * gradient2;
    
    arma::vec delta_theta = (sqrt(E_delta_theta2 + epsilon) / sqrt(E_g2 + epsilon)) % gradient;
    E_delta_theta2 = rho * E_delta_theta2 + (1 - rho) * (delta_theta % delta_theta);
    theta -= delta_theta;
    
    prev_theta = theta;
      
    if (arma::norm(gradient, 2) < epsilon) break; 
  }
  return theta;
}