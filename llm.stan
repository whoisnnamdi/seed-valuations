data {
  int<lower=1> n;
  vector[n] y;
}

parameters {
  vector[n] mu;
  real<lower=0> sigma_level;
  real<lower=0> sigma_irreg;
}

transformed parameters {
  vector[n] yhat;
  yhat = mu;
}

model {
  for(t in 2:n)
    mu[t] ~ normal(mu[t-1], sigma_level);
    y ~ normal(yhat, sigma_irreg);
}

generated quantities {
  vector[n] log_lik;
  vector[n] yrep;
  for (t in 1:n) {
    log_lik[t] = normal_lpdf(y[t] | yhat[t], sigma_irreg);
    yrep[t] = normal_rng(yhat[t], sigma_irreg);
  }
}
