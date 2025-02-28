#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

mat add_column(mat A, vec v) {
  A.shed_col(0);
  return join_rows(A,v);
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

arma::vec adasmooth_logistic(const arma::mat& X, const arma::vec& y, double eta = 0.1, 
                           int max_iter = 1000, double epsilon = 1e-6,
                           double rho1 = 0.5, double rho2 = 0.99, int M = 10) {
  int n = X.n_rows;
  int p = X.n_cols;
  arma::vec theta = arma::zeros(p);
  arma::vec E_g2 = zeros(p); 
  arma::vec prev_theta = theta;
  arma::mat delta_theta = zeros(p,M);
  
  arma::vec et = zeros(p);
  
  arma::vec preds = zeros(p);
  arma::vec gradient = zeros(p);
  
  for (int iter = 0; iter < max_iter; iter++) {
    delta_theta = zeros(p,M);
    for (int batch = 0; batch < M; batch++){
      preds = 1.0 / (1.0 + arma::exp(-X * theta)); 
      gradient = X.t() * (preds - y) / n; 
      if(batch != 0){
        et = abs(sum(delta_theta, 1))/sum(abs(delta_theta), 1);
      }
      arma::vec ct = (rho2 - rho1) * et + (1 - rho2);
      arma::vec ct2 = ct % ct;
      arma::vec gradient2 = gradient % gradient;
      
      E_g2 = ct2 % gradient2 + (1 - ct2) % E_g2;
      
      theta -= (eta / sqrt(E_g2 + epsilon)) % gradient;
      
      delta_theta = add_column(delta_theta, -(eta / (sqrt(E_g2) + epsilon)) % gradient);
      prev_theta = theta;
      
      if (arma::norm(gradient, 2) < epsilon) break; 
    }
    if (arma::norm(gradient, 2) < epsilon) break; 
  }
  return theta;
}