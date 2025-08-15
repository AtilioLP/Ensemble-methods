import pandas as pd
import numpy as np
from numpy import log, exp
from scipy.stats import lognorm
from scipy.optimize import minimize
from matplotlib import use, get_backend
use('TkAgg', force=True)
from matplotlib import pyplot as plt
print("Switched to:", get_backend())

import aux_functions
import pyLogPool

raw_preds = pd.read_csv("data/tbl.sprint.uf.week.train3.csv")

k = 1324-1 ##peak weak (in São Paulo)
l_pos = tuple([col for col in raw_preds.columns if 'lower' in col])
u_pos = tuple([col for col in raw_preds.columns if 'upper' in col])
the_pos = list(l_pos + u_pos)
l_pos = list(l_pos)
u_pos = list(u_pos)
omega = raw_preds["cases"][k-1]

lvls = (0.5, 0.8, 0.9, 0.95)
p = [aux_functions.quantile_pair(lvl) for lvl in lvls]
p = [i for quantile in p for i in quantile]
p.append(0.5)
ps = p.copy()
ps.sort()


xi_values = list(raw_preds.loc[k, the_pos].values)
xi_values.append(raw_preds.loc[k, 'pred'])
xi_sorted = xi_values.copy()
xi_sorted.sort()

out = pd.DataFrame({
    'p': ps,
    'xi': xi_sorted
})

## Get parametric approximations

lowers = list(raw_preds.loc[k, l_pos])
J = len(lowers)
names_l = [name.split('_') for name in raw_preds.loc[k, l_pos].index]
extracted_lvls = [float(parts[1]) for parts in names_l if len(parts) > 1]

uppers = list(raw_preds.loc[k, u_pos])

intervals = pd.DataFrame({
    'level': [i / 100 for i in extracted_lvls],
    'lwr': lowers,
    'med': [raw_preds.loc[k, 'pred']] * J,
    'upr': uppers
})

approxs = [[], []]
for j in range(0, J):
    aux = aux_functions.get_lognormal_pars(med = intervals['med'][j],
                             lwr = intervals['lwr'][j],
                             upr = intervals['upr'][j],
                             alpha = intervals['level'][j])
    approxs[0].append(aux[0])
    approxs[1].append(aux[1])

ln_approx = pd.DataFrame({
    'level': list(intervals['level']),
    'mu': approxs[0],
    'sigma': approxs[1]
})

############### HARMONISATION ###############

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

cdf_fit = fit_ln_CDF(x = out['xi'],
                     Fhat = out['p'],
                     weighting = 1)

fitted_cdf = lognorm.cdf(out['xi'], s=cdf_fit[1], scale=np.exp(cdf_fit[0]))

plt.plot(out['xi'], out['p'], 'o',  markerfacecolor = 'none', markeredgecolor = 'black')
plt.plot(out['xi'], fitted_cdf, color = 'black', linewidth=1.5)
plt.xlabel('xi')
plt.ylabel('CDF')
plt.show()

### Find log-normal which minimises the sum of KLs
def opt_fn1(par):
    kls = np.full(J, np.nan)
    for j in range(0,J):
        kls[j] = aux_functions.kl_lognormal(mu1 = par[0],
                                            sigma1 = np.exp(par[1]),
                                            mu2 = ln_approx['mu'][j],
                                            sigma2 = ln_approx['sigma'][j])

    return np.sum(kls)

Opt_direct_KL = minimize(opt_fn1,
                         x0 = np.array([0.0, 0.0]),
                         method = 'L-BFGS-B')

direct_KL_pars = [Opt_direct_KL.x[0], exp(Opt_direct_KL.x[1])]


### Find the alpha which minimises the sum of KLs (Log-pooling)
def get_lognormal_pool_pars(ms, vs, weights):
    pars = pyLogPool.pool_par_gauss(alpha = weights, m = ms, v = vs)
    return pars

def opt_fn2(par):
    kls = np.full(J, np.nan)
    ws = pyLogPool.alpha_01(par)
    pool = get_lognormal_pool_pars(ms = ln_approx['mu'],
                                   vs = ln_approx['sigma']**2,
                                   weights = ws)
    for j in range(0,J):
        kls[j] = aux_functions.kl_lognormal(mu1 = pool[0],
                                            sigma1 = np.sqrt(pool[1]),
                                            mu2 = ln_approx['mu'][j],
                                            sigma2 = ln_approx['sigma'][j])

    return np.sum(kls)

Opt_LP_KL = minimize(opt_fn2,
                     x0 = pyLogPool.alpha_real(np.full(J, 1 / J)))

opt_LP_alpha = pyLogPool.alpha_01(Opt_LP_KL.x)

opt_LP_KL_pars = get_lognormal_pool_pars(ln_approx['mu'],
                                     ln_approx['sigma']**2,
                                     weights = opt_LP_alpha)

LP_KL_pars = [opt_LP_KL_pars[0], opt_LP_KL_pars[1]]

#### Results

print(direct_KL_pars)
print(LP_KL_pars)
print(cdf_fit)

eps = 1e-2

minX = min([lognorm.ppf(eps, s=direct_KL_pars[1], scale=np.exp(direct_KL_pars[0])),
            lognorm.ppf(eps, s=LP_KL_pars[1], scale=np.exp(LP_KL_pars[0])),
            lognorm.ppf(eps, s=cdf_fit[1], scale=np.exp(cdf_fit[0]))
            ])

maxX = max([lognorm.ppf(1 - eps, s=direct_KL_pars[1], scale=np.exp(direct_KL_pars[0])),
            lognorm.ppf(1 - eps, s=LP_KL_pars[1], scale=np.exp(LP_KL_pars[0])),
            lognorm.ppf(1 - eps, s=cdf_fit[1], scale=np.exp(cdf_fit[0]))
            ])

xs = np.linspace(minX, maxX, num=1000)

individual_densities = []
for j in range(0, J):
    mu_j = ln_approx['mu'][j]
    sigma_j = ln_approx['sigma'][j]
    level_j = ln_approx['level'][j]

    density = lognorm.pdf(xs, s = sigma_j, scale = np.exp(mu_j))
    df = pd.DataFrame({
        'x': xs,
        'dens': density,
        'level': str(level_j)
    })
    individual_densities.append(df)

Individual = pd.concat(individual_densities, ignore_index=True)

pooled_densities = [
    pd.DataFrame({
        'x': xs,
        'dens': lognorm.pdf(xs,
                            s = cdf_fit[1],
                            scale = np.exp(cdf_fit[0])),
        'dist': 'pool_CDF'
    }),
    pd.DataFrame({
        'x': xs,
        'dens': lognorm.pdf(xs,
                            s = direct_KL_pars[1],
                            scale = np.exp(direct_KL_pars[0])),
        'dist': 'pool_minKL'
    }),
    pd.DataFrame({
        'x': xs,
        'dens': lognorm.pdf(xs,
                            s = LP_KL_pars[1],
                            scale = np.exp(LP_KL_pars[0])),
        'dist': 'pool_minKL-LP'
    })
]

Pooled = pd.concat(pooled_densities, ignore_index=True)

#Plot
fig, ax = plt.subplots(figsize=(10, 6))

for dist in Pooled['dist'].unique():
    subset = Pooled[Pooled['dist'] == dist]
    ax.plot(subset['x'], subset['dens'], label=dist, linewidth=2.5)

for level in Individual['level'].unique():
    subset = Individual[Individual['level'] == level]
    ax.plot(subset['x'], subset['dens'], label=level, linestyle='--', linewidth=2.0)

ax.axvline(x=raw_preds.loc[k, 'cases'], linestyle='dashed', color='black')
x_ticks = np.arange(0, 125001, 12500)
ax.set_xticks(x_ticks)
ax.set_xlabel("Cases")
y_ticks = np.arange(0, 0.00012501, 0.00001250)
ax.set_yticks(y_ticks)
ax.set_ylabel("Density")
ax.set_title(f"{raw_preds.loc[k, 'uf']} {raw_preds.loc[k, 'date']}")
ax.margins(x=0, y=0)

fig.tight_layout()
plt.show()