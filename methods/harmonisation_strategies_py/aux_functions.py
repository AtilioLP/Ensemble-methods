import numpy as np
from numpy import log
from numpy import exp
from scipy.stats import lognorm
from scipy.optimize import minimize

import pyLogPool

def quantile_pair(p):
    return ((1-p)/2, (1+p)/2)

def get_lognormal_pars(med, lwr, upr, alpha = 0.95):
    def loss(theta):
        tent_qs = lognorm.ppf([(1 - alpha) / 2, (1 + alpha) / 2],
                              scale = exp(theta[0]),  # scale = exp(meanlog)
                              s = exp(theta[1]))  # s = sdlog
        if (lwr == 0):
            return abs(upr - tent_qs[1]) / upr
        else:
            return abs(lwr - tent_qs[0]) / lwr + abs(upr - tent_qs[1]) / upr

    mustar = log(med)
    bounds = [(-5 * abs(mustar), 5 * abs(mustar)), (None, log(10))]

    Opt = minimize(loss,
                   x0 = [mustar, 1/2],
                   method='L-BFGS-B',
                   bounds=bounds)
    meanlog_opt, log_sdlog_opt = Opt.x
    return [meanlog_opt, exp(log_sdlog_opt)]

def kl_lognormal(mu1, sigma1, mu2, sigma2):
    term1 = log(sigma2 / sigma1)
    term2 = (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2)
    kl = term1 + term2 - 0.5
    return kl

### Direct estimation from the ECDF
def fit_ln_CDF(x, Fhat, weighting = 1):
    K = len(x)
    if len(Fhat) != K:
        raise ValueError("Size mismatch between x and Fhat.")

    log_probs = log(Fhat)

    if weighting == 1:
        weights = 1 / (Fhat * (1 - Fhat)) * 1 / np.abs(log_probs)
    else:
        weights = np.ones(K)

    def opt_cdf_diff(par, ws = weights):
        mu = par[0]
        sigma = np.exp(par[1])
        theo_logF = lognorm.logcdf(x,
                                   s = sigma,
                                   scale = np.exp(mu))
        loss = np.sum(ws * np.abs(log_probs - theo_logF))
        return loss

    Opt = minimize(opt_cdf_diff,
                   x0 = np.array([np.mean(np.log(x)), 0]),
                   method = 'L-BFGS-B')

    return [Opt.x[0], np.exp(Opt.x[1])]

### Find log-normal which minimises the sum of KLs
def minimize_opt_fn1(method, x0, J, ln_approx):
    def opt_fn1(par):
        kls = np.full(J, np.nan)
        for j in range(0, J):
            kls[j] = kl_lognormal(mu1 = par[0],
                                  sigma1 = np.exp(par[1]),
                                  mu2 = ln_approx['mu'][j],
                                  sigma2 = ln_approx['sigma'][j])

        return np.sum(kls)

    result_opt_fn1 = minimize(opt_fn1,
                              x0=x0,
                              method=method)
    return result_opt_fn1

### Find the alpha which minimises the sum of KLs (Log-pooling)
def get_lognormal_pool_pars(ms, vs, weights):
    pars = pyLogPool.pool_par_gauss(alpha = weights, m = ms, v = vs)
    return pars

def minimize_opt_fn2(x0, J, ln_approx):

    def opt_fn2(par):
        kls = np.full(J, np.nan)
        ws = pyLogPool.alpha_01(par)
        pool = get_lognormal_pool_pars(ms = ln_approx['mu'],
                                       vs = ln_approx['sigma']**2,
                                       weights = ws)
        for j in range(0,J):
            kls[j] = kl_lognormal(mu1 = pool[0],
                                  sigma1 = pool[1],
                                  mu2 = ln_approx['mu'][j],
                                  sigma2 = ln_approx['sigma'][j])

        return np.sum(kls)

    result_opt_fn2 = minimize(opt_fn2,
                              x0=x0)

    return result_opt_fn2
