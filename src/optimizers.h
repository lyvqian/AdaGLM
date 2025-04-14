#ifndef OPTIMIZERSH
#define OPTIMIZERSH

#include <RcppArmadillo.h>
#include "families.h"

enum OptimizerType { ADAM, AdaGrad, AdaSmooth, AdaDelta };

struct OptimizerConfig {
  double alpha = 0.01;
  int max_iter = 1000;
  double epsilon = 1e-6;
  OptimizerType method = ADAM;
  
  // Adam-specific
  double beta1 = 0.9;
  double beta2 = 0.999;
  
  // AdaDelta-specific 
  double rho = 0.95;
  
  //AdaSmooth-specific
  double rho1 = 0.5;
  double rho2 = 0.99;
  int M = 10;
};

arma::vec run_optimizer(const arma::mat& X, const arma::vec& y,
                        Family* family, const OptimizerConfig& cfg) {
  int n = X.n_rows;
  int p = X.n_cols;
  arma::vec theta = arma::zeros(p); 
  
  arma::vec m = arma::zeros(p); // for ADAM
  arma::vec v = arma::zeros(p); // for ADAM
  
  arma::vec g_squared = arma::zeros(p);  // for AdaGrad
  
  arma::vec E_g2 = arma::zeros(p); // for AdaDelta and AdaSmooth
  arma::vec E_delta_theta2 = arma::zeros(p); // for AdaDelta
  arma::vec deltaX2 = arma::zeros(p); // for AdaDelta
  
  arma::vec et = arma::zeros(p); // for AdaSmooth
  
  for (int iter = 0; iter < cfg.max_iter; iter++) {
    arma::vec eta = X * theta;
    arma::vec mu = family->inverse_link(eta);
    arma::vec residual = y - mu;
    arma::vec grad = X.t() * (mu - y) / n;
    
    arma::vec update;
    if (cfg.method == ADAM) {
      m = cfg.beta1 * m + (1 - cfg.beta1) * grad;
      v = cfg.beta2 * v + (1 - cfg.beta2) * arma::square(grad);
      
      arma::vec m_hat = m / (1 - std::pow(cfg.beta1, iter+1));
      arma::vec v_hat = v / (1 - std::pow(cfg.beta2, iter+1));
      
      update = cfg.alpha * m_hat / (arma::sqrt(v_hat) + 1e-8);
      
      arma::vec theta_old = theta;
      theta -= update;
      
      if (arma::norm(theta - theta_old, 2) < cfg.epsilon)
        break;
      
    } else if (cfg.method == AdaGrad) { 
      g_squared += arma::square(grad);
      arma::vec adjusted_alpha = cfg.alpha / (arma::sqrt(g_squared) + cfg.epsilon);
      
      update = adjusted_alpha % grad;
      
      arma::vec theta_old = theta;
      theta -= update;
      
      if (arma::norm(theta - theta_old, 2) < cfg.epsilon)
        break;
      
    } else if (cfg.method == AdaDelta){
      arma::vec gradient2 = grad % grad;
      E_g2 = cfg.rho * E_g2 + (1 - cfg.rho) * gradient2;
      arma::vec delta_theta = (sqrt(E_delta_theta2 + 1e-8) / sqrt(E_g2 + 1e-8)) % grad;
      E_delta_theta2 = cfg.rho * E_delta_theta2 + (1 - cfg.rho) * (delta_theta % delta_theta);
      
      update = delta_theta;
      arma::vec theta_old = theta;
      theta -= update;
      
      if (arma::norm(theta - theta_old, 2) < cfg.epsilon)
        break;
      
    } else if (cfg.method == AdaSmooth){
      arma::mat delta_theta = arma::zeros(p, cfg.M);
      for (int batch = 0; batch < cfg.M; batch++){
        arma::vec eta = X * theta;
        arma::vec mu = family->inverse_link(eta);
        arma::vec grad = X.t() * (mu - y) / n;
        if(batch != 0){
          et = abs(sum(delta_theta, 1))/sum(abs(delta_theta), 1);
        }
        arma::vec ct = (cfg.rho2 - cfg.rho1) * et + (1 - cfg.rho2);
        arma::vec ct2 = ct % ct;
        arma::vec gradient2 = grad % grad;

        E_g2 = ct2 % gradient2 + (1 - ct2) % E_g2;
        
        update = (cfg.alpha / (arma::sqrt(E_g2) + 1e-6)) % grad;

        delta_theta.shed_col(0);
        delta_theta = join_rows(delta_theta,update);
        
        arma::vec theta_old = theta;
        theta -= update;

        if (arma::norm(theta-theta_old, 2) < cfg.epsilon) break;
      }
    }

    
  }
  
  return theta;
}

#endif
