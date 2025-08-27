import pandas as pd
import numpy as np
from numpy import exp

import scoringrules as sr

import aux_functions

df_cases = pd.read_csv("data/dengue_uf.csv.gz", compression = "gzip")

df_pred = pd.read_csv("data/preds_2nd_sprint.csv.gz", compression = "gzip")

print(df_cases.head(10))
print(df_pred.head(10))

max_date = min(max(df_pred['date']), max(df_cases['date']))
min_date = max(min(df_pred['date']), min(df_cases['date']))

df_pred = df_pred[(df_pred['date'] >= min_date) & (df_pred['date'] <= max_date)]
df_cases = df_cases[(df_cases['date'] >= min_date) & (df_cases['date'] <= max_date)]
df_pred = df_pred.rename(columns={'state': 'uf'})

filtered_df_cases = df_cases[df_cases[['date', 'uf']].apply(tuple, axis=1).isin(df_pred[['date', 'uf']].apply(tuple, axis=1))]

merged_df = df_pred.merge(filtered_df_cases[['date', 'uf'] + ['epiweek', 'casos']], on=['date', 'uf'], how='left')

merged_df['casos'] = merged_df['casos'].fillna(0)

# Filtering for a state (uf)
merged_df = merged_df[merged_df['uf'].isin(['RJ'])]
merged_df.reset_index(drop=True, inplace=True)

print(len(merged_df['casos']))

#1 - Log normal using cdf.
merged_df['met1_location'] = np.nan
merged_df['met1_scale'] = np.nan
#2 - Log normal minimizing KL distance.
merged_df['met2_location'] = np.nan
merged_df['met2_scale'] = np.nan


l_pos = tuple([col for col in merged_df.columns if 'lower' in col])
u_pos = tuple([col for col in merged_df.columns if 'upper' in col])
the_pos = list(l_pos + u_pos)
l_pos = list(l_pos)
u_pos = list(u_pos)

lvls = (0.5, 0.8, 0.9, 0.95)
p = [aux_functions.quantile_pair(lvl) for lvl in lvls]
p = [i for quantile in p for i in quantile]
p.append(0.5)
ps = p.copy()
ps.sort()


def estimate_lognormals(k):
    xi_values = list(merged_df.loc[k, the_pos].values)
    xi_values.append(merged_df.loc[k, 'pred'])
    xi_sorted = xi_values.copy()
    xi_sorted.sort()

    out = pd.DataFrame({
        'p': ps,
        'xi': xi_sorted
    })

    lowers = list(merged_df.loc[k, l_pos])
    J = len(lowers)
    names_l = [name.split('_') for name in merged_df.loc[k, l_pos].index]
    extracted_lvls = [float(parts[1]) for parts in names_l if len(parts) > 1]

    uppers = list(merged_df.loc[k, u_pos])

    intervals = pd.DataFrame({
        'level': [i / 100 for i in extracted_lvls],
        'lwr': lowers,
        'med': [merged_df.loc[k, 'pred']] * J,
        'upr': uppers
    })

    approxs = [[], []]
    for j in range(0, J):
        aux = aux_functions.get_lognormal_pars(med=intervals['med'][j],
                                               lwr=intervals['lwr'][j],
                                               upr=intervals['upr'][j],
                                               alpha=intervals['level'][j])
        approxs[0].append(aux[0])
        approxs[1].append(aux[1])

    ln_approx = pd.DataFrame({
        'level': list(intervals['level']),
        'mu': approxs[0],
        'sigma': approxs[1]
    })

    cdf_fit = aux_functions.fit_ln_CDF(x=out['xi'],
                                       Fhat=out['p'],
                                       weighting=1)

    Opt_direct_KL = aux_functions.minimize_opt_fn1(method='L-BFGS-B',
                                                   x0=np.array([0.0, 0.0]),
                                                   J=J,
                                                   ln_approx=ln_approx)

    #direct_KL_pars = [Opt_direct_KL.x[0], exp(Opt_direct_KL.x[1])]

    return [cdf_fit, Opt_direct_KL.x[0], exp(Opt_direct_KL.x[1])]

met1_loc = np.empty(len(merged_df['casos']))
met1_scale = np.empty(len(merged_df['casos']))
met2_loc = np.empty(len(merged_df['casos']))
met2_scale = np.empty(len(merged_df['casos']))

#merged_df[["lower_95", "lower_90", "lower_80", "lower_50"]] = merged_df[["lower_95", "lower_90", "lower_80", "lower_50"]].clip(lower=1e-16)

for i in range(0, len(merged_df['casos'])):
    aux = estimate_lognormals(k=i)
    met1_loc[i] = aux[0][0]
    met1_scale[i] = aux[0][1]
    met2_loc[i] = aux[1]
    met2_scale[i] = aux[2]
    if i % 100 == 0:
        print(i)

merged_df['met1_location'] = met1_loc
merged_df['met1_scale'] = met1_scale
merged_df['met2_location'] = met2_loc
merged_df['met2_scale'] = met2_scale

#calculating CRPS
aux = np.zeros([len(merged_df['casos']), 2])

for i in range(0, len(merged_df['casos'])):
    aux[i, 0] = sr.crps_lognormal(merged_df['casos'][i], merged_df['met1_location'][i], merged_df['met1_scale'][i])
    aux[i, 1] = sr.crps_lognormal(merged_df['casos'][i], merged_df['met2_location'][i], merged_df['met2_scale'][i])

merged_df2 = merged_df.copy()

merged_df2['met1_CRPS'] = aux[:, 0]
merged_df2['met2_CRPS'] = aux[:, 1]

merged_df2.to_csv('data/merged_df_RJ.csv', index=False)
