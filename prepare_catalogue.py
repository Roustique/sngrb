import numpy as np
import pandas as pd
from scipy.stats import norm

cat0 = pd.read_excel('./catalogues/work_catalog_2022.xlsx', usecols = 'A,C:H,N:O,X')
cat0 = cat0.replace('N', np.nan)
cat0 = cat0.replace(0, np.nan)
cat0 = cat0.replace('>100', 100)
cat0 = cat0.replace('>300', 300)

# Replacing unknown uncertanties with median relative uncertanties
def replace_nan_by_med(cat, colnames, sym=False):
    if sym:
        col, err = colnames
        mederr = np.median(cat[err][~np.isnan(cat[err])] / cat[col][~np.isnan(cat[err])])
        cat.loc[np.isnan(cat[err]), err] = mederr * cat[col][np.isnan(cat[err])]
    else:
        col, toplim, botlim = colnames
        toperr = ( cat[toplim][~np.isnan(cat[toplim])] - cat[col][~np.isnan(cat[toplim])] ) / cat[col][~np.isnan(cat[toplim])]
        boterr = ( cat[col][~np.isnan(cat[botlim])] - cat[botlim][~np.isnan(cat[botlim])] ) / cat[col][~np.isnan(cat[botlim])]
        topmed = np.median(toperr)
        botmed = np.median(boterr)
        cat.loc[np.isnan(cat[toplim]), toplim] = cat[col][np.isnan(cat[toplim])] * (1 + topmed)
        cat.loc[np.isnan(cat[botlim]), botlim] = cat[col][np.isnan(cat[botlim])] * (1 - botmed)

replace_nan_by_med(cat0, ('CPL:alpha', 'CPL:alpha+', 'CPL:alpha-'))
replace_nan_by_med(cat0, ('CPL:Ep', 'CPL:Ep+', 'CPL:Ep-'))
replace_nan_by_med(cat0, ('BAT Fluence\n(15-150 keV)\n[10-7 erg/cm2]', 'BAT Fluence\n90% Error\n(15-150 keV)\n[10-7 erg/cm2]'), sym=True)

cat0.rename(columns={
    'CPL:alpha': 'alpha',
    'CPL:alpha+': 'alpha_u',
    'CPL:alpha-': 'alpha_d',
    'CPL:Ep': 'e_p',
    'CPL:Ep+': 'e_p_u',
    'CPL:Ep-': 'e_p_d',
    'BAT Fluence\n(15-150 keV)\n[10-7 erg/cm2]': 's_obs',
    'BAT Fluence\n90% Error\n(15-150 keV)\n[10-7 erg/cm2]': 'd_s_obs_90',
    'Redshift': 'z'
}, inplace=True)

cat0['d_alpha_u'] = cat0['alpha_u'] - cat0['alpha']
cat0['d_alpha_d'] = cat0['alpha'] - cat0['alpha_d']
cat0['d_e_p_u'] = cat0['e_p_u'] - cat0['e_p']
cat0['d_e_p_d'] = cat0['e_p'] - cat0['e_p_d']
cat0['d_s_obs'] = cat0['d_s_obs_90'] / norm.ppf(0.95)

nonans_mask = ~cat0['s_obs'].isna() & (cat0['d_alpha_u'] > 0)
cat0 = cat0.loc[nonans_mask].reset_index(drop=True)
cat0 = cat0[[
    'GRB',
    'z',
    'alpha',
    'd_alpha_u',
    'd_alpha_d',
    'e_p',
    'd_e_p_u',
    'd_e_p_d',
    's_obs',
    'd_s_obs'
]]

cat0.to_csv('catalogues/work_catalog_2022_prepared.csv')