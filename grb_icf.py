import os
import numpy as np
from scipy.optimize import least_squares
from scipy.stats.mstats import theilslopes
import pandas as pd
from joblib import Parallel, delayed, cpu_count
import warnings
from sngrb_utils import cosmology, sampling, plotting

warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore', over='ignore')

sample_size = 1000  # Monte-Carlo sample size
n_threads = cpu_count()

# Reading data from prepared catalogue
prepared_cat_name = 'catalogues/work_catalog_2022_prepared.csv'
if not os.path.isfile(prepared_cat_name):
    raise FileExistsError('Prepared catalogue not found. Run "python3 prepare_catalogue.py" first.')
cat = pd.read_csv(prepared_cat_name)
grb_names_arr = cat['GRB'].values
z_arr = cat['z'].values
alpha_arr = cat['alpha'].values
d_alpha_u_arr = cat['d_alpha_u'].values
d_alpha_d_arr = cat['d_alpha_d'].values
e_p_arr = cat['e_p'].values
d_e_p_u_arr = cat['d_e_p_u'].values
d_e_p_d_arr = cat['d_e_p_d'].values
s_obs_arr = cat['s_obs'].values
d_s_obs_arr = cat['d_s_obs'].values
grb_amount = len(grb_names_arr)


def loopfun_makesamples(i):
    alpha_sample, e_p_sample, s_obs_sample = sampling.make_samples(i, (
        alpha_arr,
        d_alpha_d_arr,
        d_alpha_u_arr,
        e_p_arr,
        d_e_p_d_arr,
        d_e_p_u_arr,
        s_obs_arr,
        d_s_obs_arr,
        z_arr
    ), amount=sample_size)
    amx_sample, amy_sample = sampling.sample_amatixy(i, (alpha_sample, e_p_sample, s_obs_sample, z_arr))
    return alpha_sample, e_p_sample, s_obs_sample, amx_sample, amy_sample


print('Creating MC samples for GRBs')
alpha_all_samples, e_p_all_samples, s_obs_all_samples, amx_all_samples, amy_all_samples = np.array(list(
    zip(*Parallel(n_jobs=n_threads, max_nbytes='2048M', verbose=10)(
        delayed(loopfun_makesamples)(i) for i in np.arange(grb_amount)))))

amx_meds, amx_dlim, amx_ulim = sampling.get_meds_and_lims(amx_all_samples)
amy_meds, amy_dlim, amy_ulim = sampling.get_meds_and_lims(amy_all_samples)
amx_flat = amx_all_samples.flatten()
amy_flat = amy_all_samples.flatten()

# Run the Theil-Sen approximation for each experiment in samples
a_sample = np.empty(sample_size)
b_sample = np.empty(sample_size)
for i in np.arange(sample_size):
    finitemask = np.isfinite(amx_all_samples[:, i]) * np.isfinite(amy_all_samples[:, i])
    fit_res = theilslopes(amy_all_samples[:, i][finitemask], amx_all_samples[:, i][finitemask])
    a_sample[i] = fit_res[0]
    b_sample[i] = fit_res[1]

a_est = np.median(a_sample)
b_est = np.median(b_sample)
d_a_est = (np.percentile(a_sample, 100 * sampling.ERR_QU) - np.percentile(a_sample, 100 * sampling.ERR_QL)) / 2
d_b_est = (np.percentile(b_sample, 100 * sampling.ERR_QU) - np.percentile(b_sample, 100 * sampling.ERR_QL)) / 2

if not os.path.isdir('pics'):
    os.mkdir('pics')

plotting.plot_corner('cornerplot_ts', {r'$a$': a_sample, r'$b$': b_sample})

finitemask = np.isfinite(amx_flat) * np.isfinite(amy_flat)
amx_flat = amx_flat[finitemask]
amy_flat = amy_flat[finitemask]

plotting.plot_amati(
    (a_sample, b_sample, amx_flat, amy_flat, amx_meds, amy_meds),
    'Amati_ts',
    'repeated Theil-Sen estimation',
    plot_errbars=False,
    col2='darkblue')
plotting.plot_amati(
    (a_sample, b_sample, amx_meds, amy_meds, amx_ulim, amy_ulim, amx_dlim, amy_dlim),
    'Amati_ts',
    'repeated Theil-Sen estimation',
    plot_samples=False,
    col1='teal')
plotting.plot_amati(
    (a_sample, b_sample, amx_flat, amy_flat, amx_meds, amy_meds, amx_ulim, amy_ulim, amx_dlim, amy_dlim),
    'Amati_ts',
    'repeated Theil-Sen estimation')


def loopfun_sample_mu_A(i):
    return tuple(
        sampling.sample_mu_a(
            i,
            (alpha_all_samples[i, :], e_p_all_samples[i, :], s_obs_all_samples[i, :], z_arr),
            a_sample,
            b_sample,
            0.0))


print('Calculating mu_A for sample experiments with Theil-Sen estimated parameters a and b')
mu_a_all_samples = np.array(list(zip(*Parallel(n_jobs=n_threads, max_nbytes='2048M', verbose=10)(
    delayed(loopfun_sample_mu_A)(i) for i in np.arange(grb_amount))))).T

mu_a_meds, mu_a_dlim, mu_a_ulim = sampling.get_meds_and_lims(mu_a_all_samples)
plotting.plot_hd('HD_ts', z_arr, mu_a_meds, mu_a_dlim, mu_a_ulim)


def lcdm_inv_residuals(pars, z_arg, alpha_arg, E_p_arg, S_obs_arg, Amx_arg, Amy_arg):
    mu_a_inner = cosmology.mu_a_vec(z_arg, alpha_arg, E_p_arg, S_obs_arg, pars[0], pars[1], pars[2])
    infinitemask_inner = ~np.isfinite(mu_a_inner)
    mu_a_inner[infinitemask_inner] = 0.0
    Amy_arg[~np.isfinite(Amy_arg)] = 0.0
    Amx_arg[~np.isfinite(Amx_arg)] = 0.0
    residuals_hd = (mu_a_inner - cosmology.mu_cosm_vec(z_arg))
    residuals_am = (Amy_arg - pars[0] * Amx_arg - pars[
        1]) * 2.2  # Multiplying by 2.2 to make residuals have similar scale
    return np.concatenate((residuals_hd, residuals_am))


def loopfun_icf(i):
    return tuple(least_squares(lcdm_inv_residuals, (1.0, 50.0, 0.0), loss='soft_l1', f_scale=2.5, args=(
        z_arr, alpha_all_samples[:, i], e_p_all_samples[:, i], s_obs_all_samples[:, i], amx_all_samples[:, i],
        amy_all_samples[:, i])).x)


print('Calculating a, b and k using icf with robust least squares')
a_inv_sample, b_inv_sample, k_inv_sample = np.array(
    list(zip(*Parallel(n_jobs=n_threads, max_nbytes='2048M', verbose=10)(
        delayed(loopfun_icf)(i) for i in np.arange(sample_size)))))

plotting.plot_corner('cornerplot_icf', {r'$a$': a_inv_sample, r'$b$': b_inv_sample, r'$k$': k_inv_sample})

a_inv_est = np.median(a_inv_sample)
Da_inv = (np.percentile(a_inv_sample, 100 * sampling.ERR_QU) - np.percentile(a_inv_sample, 100 * sampling.ERR_QL)) / 2
b_inv_est = np.median(b_inv_sample)
Db_inv = (np.percentile(b_inv_sample, 100 * sampling.ERR_QU) - np.percentile(b_inv_sample, 100 * sampling.ERR_QL)) / 2
k_inv_est = np.median(k_inv_sample)
Dk_inv = (np.percentile(k_inv_sample, 100 * sampling.ERR_QU) - np.percentile(k_inv_sample, 100 * sampling.ERR_QL)) / 2


def lcdm_inv_residuals_k0(pars, z_arg, alpha_arg, e_p_arg, s_obs_arg, amx_arg, amy_arg):
    mu_a_inner = cosmology.mu_a_vec(z_arg, alpha_arg, e_p_arg, s_obs_arg, pars[0], pars[1], 0.0)
    infinitemask_inner = ~np.isfinite(mu_a_inner)
    mu_a_inner[infinitemask_inner] = 0.0
    amy_arg[~np.isfinite(amy_arg)] = 0.0
    amx_arg[~np.isfinite(amx_arg)] = 0.0
    residuals_hd = (mu_a_inner - cosmology.mu_cosm_vec(z_arg))
    residuals_am = (amy_arg - pars[0] * amx_arg - pars[
        1]) * 2.2  # Multiplying by 2.2 to make residuals have similar scale
    return np.concatenate((residuals_hd, residuals_am))


def loopfun_icf_k0(i):
    return tuple(least_squares(lcdm_inv_residuals_k0, (1.0, 50.0), loss='soft_l1', f_scale=2.5, args=(
        z_arr, alpha_all_samples[:, i], e_p_all_samples[:, i], s_obs_all_samples[:, i], amx_all_samples[:, i],
        amy_all_samples[:, i])).x)


print('Calculating a and b using icf with robust least squares, fixed k=0')
a_inv_k0_sample, b_inv_k0_sample = np.array(list(zip(*Parallel(n_jobs=n_threads, max_nbytes='2048M', verbose=10)(
    delayed(loopfun_icf_k0)(i) for i in np.arange(sample_size)))))

plotting.plot_corner('cornerplot_icf_k0', {r'$a$': a_inv_k0_sample, r'$b$': b_inv_k0_sample})

a_inv_k0_est = np.median(a_inv_k0_sample)
d_a_inv_k0 = (np.percentile(a_inv_k0_sample, 100 * sampling.ERR_QU) - np.percentile(a_inv_k0_sample,
                                                                                   100 * sampling.ERR_QL)) / 2
b_inv_k0_est = np.median(b_inv_k0_sample)
d_b_inv_k0 = (np.percentile(b_inv_k0_sample, 100 * sampling.ERR_QU) - np.percentile(b_inv_k0_sample,
                                                                                   100 * sampling.ERR_QL)) / 2


def lcdm_residuals_chi2(pars, z_arg, alpha_arg, e_p_arg, s_obs_arg, amx_arg, amy_arg):
    mu_a_inner = cosmology.mu_a_vec(z_arg, alpha_arg, e_p_arg, s_obs_arg, pars[0], pars[1], pars[2])
    infinitemask_inner = ~np.isfinite(mu_a_inner)
    mu_a_inner[infinitemask_inner] = 0.0
    amy_arg[~np.isfinite(amy_arg)] = 0.0
    amx_arg[~np.isfinite(amx_arg)] = 0.0
    residuals_hd = (mu_a_inner - cosmology.mu_cosm_vec(z_arg)) ** 2 / cosmology.mu_cosm_vec(z_arg)
    return np.sqrt(np.sum(residuals_hd))


def loopfun_sample_mu_A_inv(i):
    return tuple(
        sampling.sample_mu_a(i, (alpha_all_samples[i, :], e_p_all_samples[i, :], s_obs_all_samples[i, :], z_arr), a_inv_sample,
                    b_inv_sample, k_inv_sample))


def loopfun_sample_mu_A_inv_k0(i):
    return tuple(sampling.sample_mu_a(i, (alpha_all_samples[i, :], e_p_all_samples[i, :], s_obs_all_samples[i, :], z_arr),
                             a_inv_k0_sample, b_inv_k0_sample, 0.0))


print('Calculating mu_A for sample experiments with parameters a, b and k estimated via ICF')
mu_a_inv_all_samples = np.array(list(zip(*Parallel(n_jobs=n_threads, max_nbytes='2048M', verbose=10)(
    delayed(loopfun_sample_mu_A_inv)(i) for i in np.arange(grb_amount))))).T
print('Calculating mu_A for sample experiments with parameters a, and b estimated via ICF, fixed k=0')
mu_a_inv_k0_all_samples = np.array(list(zip(*Parallel(n_jobs=n_threads, max_nbytes='2048M', verbose=10)(
    delayed(loopfun_sample_mu_A_inv_k0)(i) for i in np.arange(grb_amount))))).T
mu_a_inv_all_samples[~np.isfinite(mu_a_inv_all_samples)] = 0.0
mu_a_inv_k0_all_samples[~np.isfinite(mu_a_inv_k0_all_samples)] = 0.0

mu_a_inv_meds, mu_a_inv_dlim, mu_a_inv_ulim = sampling.get_meds_and_lims(mu_a_inv_all_samples)
mu_a_inv_k0_meds, mu_a_inv_k0_dlim, mu_a_inv_k0_ulim = sampling.get_meds_and_lims(mu_a_inv_k0_all_samples)

plotting.plot_hd('HD_icf', z_arr, mu_a_inv_meds, mu_a_inv_dlim, mu_a_inv_ulim)
plotting.plot_hd('HD_icf_k0', z_arr, mu_a_inv_k0_meds, mu_a_inv_k0_dlim, mu_a_inv_k0_ulim)


plotting.plot_amati(
    (a_inv_sample, b_inv_sample, amx_flat, amy_flat, amx_meds, amy_meds),
    'Amati_icf',
    'inv. cosm. fitting (free $k$)',
    plot_errbars=False,
    col2='darkblue'
)
plotting.plot_amati(
    (a_inv_sample, b_inv_sample, amx_meds, amy_meds, amx_ulim, amy_ulim, amx_dlim, amy_dlim),
    'Amati_icf',
    'inv. cosm. fitting (free $k$)',
    plot_samples=False,
    col1='teal'
)
plotting.plot_amati(
    (a_inv_sample, b_inv_sample, amx_flat, amy_flat, amx_meds, amy_meds, amx_ulim, amy_ulim, amx_dlim, amy_dlim),
    'Amati_icf',
    'inv. cosm. fitting (free $k$)'
)

plotting.plot_amati(
    (a_inv_k0_sample, b_inv_k0_sample, amx_flat, amy_flat, amx_meds, amy_meds),
    'Amati_icf_k0',
    'inv. cosm. fitting ($k=0$)',
    plot_errbars=False,
    col2='darkblue'
)
plotting.plot_amati(
    (a_inv_k0_sample, b_inv_k0_sample, amx_meds, amy_meds, amx_ulim, amy_ulim, amx_dlim, amy_dlim),
    'Amati_icf_k0',
    'inv. cosm. fitting ($k=0$)',
    plot_samples=False,
    col1='teal'
)
plotting.plot_amati(
    (a_inv_k0_sample, b_inv_k0_sample, amx_flat, amy_flat, amx_meds, amy_meds, amx_ulim, amy_ulim, amx_dlim, amy_dlim),
    'Amati_icf_k0',
    'inv. cosm. fitting ($k=0$)'
)

print('Calculating chi-square for median parameters:')
print('Varying a, b, k:',
      str(lcdm_residuals_chi2((a_inv_est, b_inv_est, k_inv_est), z_arr, alpha_arr, e_p_arr, s_obs_arr, amx_meds,
                              amy_meds)))
print('Varying a, b and fixed k=0:',
      str(lcdm_residuals_chi2((a_inv_k0_est, b_inv_k0_est, 0.0), z_arr, alpha_arr, e_p_arr, s_obs_arr, amx_meds,
                              amy_meds)))
