#ifndef FAMILIESH
#define FAMILIESH

#include <RcppArmadillo.h>

class Family {
public:
  virtual arma::vec link(const arma::vec& mu) const = 0;
  virtual arma::vec inverse_link(const arma::vec& eta) const = 0;
  virtual arma::vec variance(const arma::vec& mu) const = 0;
  virtual arma::vec deviance(const arma::vec& y, const arma::vec& mu) const = 0;
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
  
  arma::vec variance(const arma::vec& mu) const override {
    return mu % (1.0 - mu);
  }
  
  arma::vec deviance(const arma::vec& y, const arma::vec& mu) const override {
    return 2 * (y % arma::log(y / mu) + (1 - y) % arma::log((1 - y) / (1 - mu)));
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
  
  arma::vec variance(const arma::vec& mu) const override {
    return arma::ones(mu.n_elem);
  }
  
  arma::vec deviance(const arma::vec& y, const arma::vec& mu) const override {
    return arma::square(y - mu);
  }
};


class GammaInverse : public Family {
public:
  arma::vec link(const arma::vec& mu) const override {
    return 1.0 / (mu + 1e-6);
  }
  
  arma::vec inverse_link(const arma::vec& eta) const override {
    return 1.0 / (eta + 1e-6);
  }
  
  arma::vec variance(const arma::vec& mu) const override {
    return arma::square(mu);
  }
  
  arma::vec deviance(const arma::vec& y, const arma::vec& mu) const override {
    return 2 * ((y - mu) / mu - arma::log(y / mu));
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
  
  arma::vec variance(const arma::vec& mu) const override {
    return mu;
  }
  
  arma::vec deviance(const arma::vec& y, const arma::vec& mu) const override {
    return 2 * (y % arma::log(y / mu) - (y - mu));
  }
};

// class InverseGaussian : public Family {
// public:
//   arma::vec link(const arma::vec& mu) const override {
//     return 1.0 / arma::square(mu);
//   }
//   
//   arma::vec inverse_link(const arma::vec& eta) const override {
//     return 1.0 / arma::sqrt(eta);
//   }
//   
//   arma::vec variance(const arma::vec& mu) const override {
//     return arma::pow(mu, 3);
//   }
//   
//   arma::vec deviance(const arma::vec& y, const arma::vec& mu) const override {
//     return (arma::square(y - mu) / (mu % mu % y));
//   }
// };


// Quasi families (identity + variance spec)

// class QuasiIdentity : public Family {
// public:
//   arma::vec link(const arma::vec& mu) const override {
//     return mu;
//   }
//   
//   arma::vec inverse_link(const arma::vec& eta) const override {
//     return eta;
//   }
//   
//   arma::vec variance(const arma::vec& mu) const override {
//     return arma::ones(mu.n_elem); // constant variance
//   }
//   
//   arma::vec deviance(const arma::vec& y, const arma::vec& mu) const override {
//     return arma::square(y - mu); // approximate
//   }
// };
// 
// class QuasiBinomialLogit : public BinomialLogit {
//   // Uses same link and inverse_link as BinomialLogit,
//   // but variance may be scaled in practice (can extend later)
// };
// 
// class QuasiPoissonLog : public PoissonLog {
//   // Same as PoissonLog but with dispersion parameter (not handled here yet)
// };




#endif
