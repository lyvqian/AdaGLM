#include <RcppArmadillo.h>
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec irwls_logistic(const arma::mat& X, const arma::vec& y, 
                         int max_iter = 100, double tol = 1e-6) {
  int n = X.n_rows;  
  int d = X.n_cols;  
  
  arma::vec w = arma::zeros(d);  
  arma::vec p(n); 
  arma::vec z(n);  
  arma::vec W(n);  
  
  for (int iter = 0; iter < max_iter; iter++) {
    p = 1 / (1 + exp(-X * w));
    
    W = p % (1 - p); 
    z = X * w + (y - p) / (W + 1e-6);  
    
    arma::mat W_diag = diagmat(W);
    
    arma::mat XtWX = X.t() * W_diag * X;
    arma::vec XtWz = X.t() * W_diag * z;
    
    arma::vec w_new = solve(XtWX, XtWz);

    w = w_new;
  }
  
  return w;
}
