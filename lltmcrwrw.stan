data {
  int<lower=1> n;
  vector[n] y;
}

parameters {
  vector[n] mu_err;
  vector[n] gamma_err;
  real<lower=0> sigma_level;
  real<lower=0> sigma_gamma;
  real<lower=0> sigma_irreg;
}

transformed parameters {
  vector[n] mu;
  vector[n] logitgamma;
  vector[n] gamma;
  mu[1] = mu_err[1];
  mu[2] = mu[1] + mu_err[2];
  logitgamma[1] = gamma_err[1];
  gamma[1] = inv_logit(logitgamma[1]);
  for (t in 2:n) {
    logitgamma[t] = logitgamma[t-1] + sigma_gamma * gamma_err[t];
    gamma[t] = inv_logit(logitgamma[t]);
    if (t > 2)
      mu[t] = mu[t-1] + gamma[t]*(mu[t-1] - mu[t-2]) + sigma_level * mu_err[t];
  }
}

model {
  sigma_level ~ exponential(0.1);
  sigma_gamma ~ exponential(0.1);
  sigma_irreg ~ exponential(0.1);

  mu_err ~ normal(0, 1);
  gamma_err ~ normal(0, 1);
  
  y ~ normal(mu, sigma_irreg);
}

generated quantities {
   vector[n] log_lik;
   for (t in 1:n) {
     log_lik[t] = normal_lpdf(y[t] | mu[t], sigma_irreg);
   }
}
