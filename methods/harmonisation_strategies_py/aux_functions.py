from numpy import log
from numpy import exp
from scipy.stats import lognorm
from scipy.optimize import minimize

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