#ifndef FAMILIESH
#define FAMILIESH

#include <RcppArmadillo.h>

class Family {
public:
  virtual arma::vec link(const arma::vec& mu) const = 0;
  virtual arma::vec inverse_link(const arma::vec& eta) const = 0;
  virtual ~Family() {}
};


class BinomialLogit : public Family {
public:
  arma::vec link(const arma::vec& mu) const override {
    return arma::log(mu / (1.0 - mu));
  }
  
  arma::vec inverse_link(const arma::vec& eta) const override {
    return 1.0 / (1.0 + arma::exp(-eta));
  }
};


class GaussianIdentity : public Family {
public:
  arma::vec link(const arma::vec& mu) const override {
    return mu;
  }
  
  arma::vec inverse_link(const arma::vec& eta) const override {
    return eta;
  }
};


class GammaLog : public Family {
public:
  arma::vec link(const arma::vec& mu) const override {
    return arma::log(mu+1e-8);  
  }
  
  arma::vec inverse_link(const arma::vec& eta) const override {
    return arma::exp(eta);  
  }
  
};


class PoissonLog : public Family {
public:
  arma::vec link(const arma::vec& mu) const override {
    return arma::log(mu);
  }
  
  arma::vec inverse_link(const arma::vec& eta) const override {
    return arma::exp(eta);
  }
  
};


#endif
