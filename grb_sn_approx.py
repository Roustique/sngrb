import os
import numpy as np
from scipy.optimize import least_squares, curve_fit
from scipy.stats.mstats import theilslopes
from scipy.stats import multivariate_normal
import pandas as pd
from matplotlib import pyplot as plt
import corner
from joblib import Parallel, delayed, cpu_count
from numba import jit
import warnings
from sngrb_utils import cosmology, sampling, plotting

warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore', over='ignore')

sample_size = 1000  # Monte-Carlo sample size
n_threads = cpu_count()

# ==============================================================================
#                         Загрузим каталог Pantheon
# ==============================================================================

catSN = pd.read_csv('catalogues/Pantheon.dat', delimiter='\t', header=0, usecols=[2, 4, 5])

z_arr_SN = np.array(catSN['zcmb'])
mu_arr_SN = np.array(catSN['mu'])  # + 19.41623729
Dmu_arr_SN = np.array(catSN['err_mu'])

z_log_range = 10 ** np.linspace(np.log10(np.min(z_arr_SN)), np.log10(np.max(z_arr_SN)))


# Определим модели:

@jit(nopython=True)
def llhood(y_data, y_model, sigma):
    return - 0.5 * np.sum(((y_data - y_model) / sigma) ** 2 + np.log(2 * pi * sigma ** 2))


@jit(nopython=True)
def polylog1(z, a0, a1):
    logz = np.log10(z)
    return a0 + a1 * logz


@jit(nopython=True)
def polylog2(z, a0, a1, a2):
    logz = np.log10(z)
    return a0 + a1 * logz + a2 * logz ** 2


@jit(nopython=True)
def polylog3(z, a0, a1, a2, a3):
    logz = np.log10(z)
    return a0 + a1 * logz + a2 * logz ** 2 + a3 * logz ** 3


@jit(nopython=True)
def polylog4(z, a0, a1, a2, a3, a4):
    logz = np.log10(z)
    return a0 + a1 * logz + a2 * logz ** 2 + a3 * logz ** 3 + a4 * logz ** 4


@jit(nopython=True)
def polylog5(z, a0, a1, a2, a3, a4, a5):
    logz = np.log10(z)
    return a0 + a1 * logz + a2 * logz ** 2 + a3 * logz ** 3 + a4 * logz ** 4 + a5 * logz ** 5


@jit(nopython=True)
def theor_just2(z, a1, a2):
    logz = np.log10(1 + z)
    return 5 * np.log10(c / H0 * z) + 25 + a1 * logz + a2 * logz ** 2


@jit(nopython=True)
def theor_just3(z, a1, a2, a3):
    logz = np.log10(1 + z)
    return 5 * np.log10(c / H0 * z) + 25 + a1 * logz + a2 * logz ** 2 + a3 * logz ** 3


@jit(nopython=True)
def theor_just4(z, a1, a2, a3, a4):
    logz = np.log10(1 + z)
    return 5 * np.log10(c / H0 * z) + 25 + a1 * logz + a2 * logz ** 2 + a3 * logz ** 3 + a4 * logz ** 4


@jit(nopython=True)
def theor_just5(z, a1, a2, a3, a4, a5):
    logz = np.log10(1 + z)
    return 5 * np.log10(c / H0 * z) + 25 + a1 * logz + a2 * logz ** 2 + a3 * logz ** 3 + a4 * logz ** 4 + a5 * logz ** 5


@jit(nopython=True)
def offpolylog2(z, a0, a1, a2):
    logz = np.log10(1 + z)
    return a0 + a1 * logz + a2 * logz ** 2


@jit(nopython=True)
def offpolylog3(z, a0, a1, a2, a3):
    logz = np.log10(1 + z)
    return a0 + a1 * logz + a2 * logz ** 2 + a3 * logz ** 3


@jit(nopython=True)
def offpolylog4(z, a0, a1, a2, a3, a4):
    logz = np.log10(1 + z)
    return a0 + a1 * logz + a2 * logz ** 2 + a3 * logz ** 3 + a4 * logz ** 4


@jit(nopython=True)
def offpolylog5(z, a0, a1, a2, a3, a4, a5):
    logz = np.log10(1 + z)
    return a0 + a1 * logz + a2 * logz ** 2 + a3 * logz ** 3 + a4 * logz ** 4 + a5 * logz ** 5


fit_pl1 = curve_fit(polylog1, z_arr_SN, mu_arr_SN, p0=np.ones(2), sigma=Dmu_arr_SN)
fit_pl2 = curve_fit(polylog2, z_arr_SN, mu_arr_SN, p0=np.ones(3), sigma=Dmu_arr_SN)
fit_pl3 = curve_fit(polylog3, z_arr_SN, mu_arr_SN, p0=np.ones(4), sigma=Dmu_arr_SN)
fit_pl4 = curve_fit(polylog4, z_arr_SN, mu_arr_SN, p0=np.ones(5), sigma=Dmu_arr_SN)
fit_pl5 = curve_fit(polylog5, z_arr_SN, mu_arr_SN, p0=np.ones(6), sigma=Dmu_arr_SN)
fit_tj2 = curve_fit(theor_just2, z_arr_SN, mu_arr_SN, p0=np.ones(2), sigma=Dmu_arr_SN)
fit_tj3 = curve_fit(theor_just3, z_arr_SN, mu_arr_SN, p0=np.ones(3), sigma=Dmu_arr_SN)
fit_tj4 = curve_fit(theor_just4, z_arr_SN, mu_arr_SN, p0=np.ones(4), sigma=Dmu_arr_SN)
fit_tj5 = curve_fit(theor_just5, z_arr_SN, mu_arr_SN, p0=np.ones(5), sigma=Dmu_arr_SN)
fit_opl2 = curve_fit(offpolylog2, z_arr_SN, mu_arr_SN, p0=np.ones(3), sigma=Dmu_arr_SN)
fit_opl3 = curve_fit(offpolylog3, z_arr_SN, mu_arr_SN, p0=np.ones(4), sigma=Dmu_arr_SN)
fit_opl4 = curve_fit(offpolylog4, z_arr_SN, mu_arr_SN, p0=np.ones(5), sigma=Dmu_arr_SN)
fit_opl5 = curve_fit(offpolylog5, z_arr_SN, mu_arr_SN, p0=np.ones(6), sigma=Dmu_arr_SN)

polylog1_best = lambda z: polylog1(z, fit_pl1[0][0], fit_pl1[0][1])
polylog2_best = lambda z: polylog2(z, fit_pl2[0][0], fit_pl2[0][1], fit_pl2[0][2])
polylog3_best = lambda z: polylog3(z, fit_pl3[0][0], fit_pl3[0][1], fit_pl3[0][2], fit_pl3[0][3])
polylog4_best = lambda z: polylog4(z, fit_pl4[0][0], fit_pl4[0][1], fit_pl4[0][2], fit_pl4[0][3], fit_pl4[0][4])
polylog5_best = lambda z: polylog5(z, fit_pl5[0][0], fit_pl5[0][1], fit_pl5[0][2], fit_pl5[0][3], fit_pl5[0][4],
                                   fit_pl5[0][5])
theor_just2_best = lambda z: theor_just2(z, fit_tj2[0][0], fit_tj2[0][1])
theor_just3_best = lambda z: theor_just3(z, fit_tj3[0][0], fit_tj3[0][1], fit_tj3[0][2])
theor_just4_best = lambda z: theor_just4(z, fit_tj4[0][0], fit_tj4[0][1], fit_tj4[0][2], fit_tj4[0][3])
theor_just5_best = lambda z: theor_just5(z, fit_tj5[0][0], fit_tj5[0][1], fit_tj5[0][2], fit_tj5[0][3], fit_tj5[0][4])
offpolylog2_best = lambda z: offpolylog2(z, fit_opl2[0][0], fit_opl2[0][1], fit_opl2[0][2])
offpolylog3_best = lambda z: offpolylog3(z, fit_opl3[0][0], fit_opl3[0][1], fit_opl3[0][2], fit_opl3[0][3])
offpolylog4_best = lambda z: offpolylog4(z, fit_opl4[0][0], fit_opl4[0][1], fit_opl4[0][2], fit_opl4[0][3],
                                         fit_opl4[0][4])
offpolylog5_best = lambda z: offpolylog5(z, fit_opl5[0][0], fit_opl5[0][1], fit_opl5[0][2], fit_opl5[0][3],
                                         fit_opl5[0][4], fit_opl5[0][5])

AIC_polylog1 = 2 * 2 - 2 * llhood(mu_arr_SN, polylog1_best(z_arr_SN), Dmu_arr_SN)
AIC_polylog2 = 2 * 3 - 2 * llhood(mu_arr_SN, polylog2_best(z_arr_SN), Dmu_arr_SN)
AIC_polylog3 = 2 * 4 - 2 * llhood(mu_arr_SN, polylog3_best(z_arr_SN), Dmu_arr_SN)
AIC_polylog4 = 2 * 5 - 2 * llhood(mu_arr_SN, polylog4_best(z_arr_SN), Dmu_arr_SN)
AIC_polylog5 = 2 * 6 - 2 * llhood(mu_arr_SN, polylog5_best(z_arr_SN), Dmu_arr_SN)
AIC_theor_just2 = 2 * 2 - 2 * llhood(mu_arr_SN, theor_just2_best(z_arr_SN), Dmu_arr_SN)
AIC_theor_just3 = 2 * 3 - 2 * llhood(mu_arr_SN, theor_just3_best(z_arr_SN), Dmu_arr_SN)
AIC_theor_just4 = 2 * 4 - 2 * llhood(mu_arr_SN, theor_just4_best(z_arr_SN), Dmu_arr_SN)
AIC_theor_just5 = 2 * 5 - 2 * llhood(mu_arr_SN, theor_just5_best(z_arr_SN), Dmu_arr_SN)
AIC_offpolylog2 = 2 * 3 - 2 * llhood(mu_arr_SN, offpolylog2_best(z_arr_SN), Dmu_arr_SN)
AIC_offpolylog3 = 2 * 4 - 2 * llhood(mu_arr_SN, offpolylog3_best(z_arr_SN), Dmu_arr_SN)
AIC_offpolylog4 = 2 * 5 - 2 * llhood(mu_arr_SN, offpolylog4_best(z_arr_SN), Dmu_arr_SN)
AIC_offpolylog5 = 2 * 6 - 2 * llhood(mu_arr_SN, offpolylog5_best(z_arr_SN), Dmu_arr_SN)

BIC_polylog1 = 2 * np.log(np.size(z_arr_SN)) - 2 * llhood(mu_arr_SN, polylog1_best(z_arr_SN), Dmu_arr_SN)
BIC_polylog2 = 3 * np.log(np.size(z_arr_SN)) - 2 * llhood(mu_arr_SN, polylog2_best(z_arr_SN), Dmu_arr_SN)
BIC_polylog3 = 4 * np.log(np.size(z_arr_SN)) - 2 * llhood(mu_arr_SN, polylog3_best(z_arr_SN), Dmu_arr_SN)
BIC_polylog4 = 5 * np.log(np.size(z_arr_SN)) - 2 * llhood(mu_arr_SN, polylog4_best(z_arr_SN), Dmu_arr_SN)
BIC_polylog5 = 6 * np.log(np.size(z_arr_SN)) - 2 * llhood(mu_arr_SN, polylog5_best(z_arr_SN), Dmu_arr_SN)
BIC_theor_just2 = 2 * np.log(np.size(z_arr_SN)) - 2 * llhood(mu_arr_SN, theor_just2_best(z_arr_SN), Dmu_arr_SN)
BIC_theor_just3 = 3 * np.log(np.size(z_arr_SN)) - 2 * llhood(mu_arr_SN, theor_just3_best(z_arr_SN), Dmu_arr_SN)
BIC_theor_just4 = 4 * np.log(np.size(z_arr_SN)) - 2 * llhood(mu_arr_SN, theor_just4_best(z_arr_SN), Dmu_arr_SN)
BIC_theor_just5 = 5 * np.log(np.size(z_arr_SN)) - 2 * llhood(mu_arr_SN, theor_just5_best(z_arr_SN), Dmu_arr_SN)
BIC_offpolylog2 = 3 * np.log(np.size(z_arr_SN)) - 2 * llhood(mu_arr_SN, offpolylog2_best(z_arr_SN), Dmu_arr_SN)
BIC_offpolylog3 = 4 * np.log(np.size(z_arr_SN)) - 2 * llhood(mu_arr_SN, offpolylog3_best(z_arr_SN), Dmu_arr_SN)
BIC_offpolylog4 = 5 * np.log(np.size(z_arr_SN)) - 2 * llhood(mu_arr_SN, offpolylog4_best(z_arr_SN), Dmu_arr_SN)
BIC_offpolylog5 = 6 * np.log(np.size(z_arr_SN)) - 2 * llhood(mu_arr_SN, offpolylog5_best(z_arr_SN), Dmu_arr_SN)

mu_range_pl1 = polylog1_best(z_log_range)
mu_range_pl2 = polylog2_best(z_log_range)
mu_range_pl3 = polylog3_best(z_log_range)
mu_range_pl4 = polylog4_best(z_log_range)
mu_range_pl5 = polylog5_best(z_log_range)

mu_range_tj2 = theor_just2_best(z_log_range)
mu_range_tj3 = theor_just3_best(z_log_range)
mu_range_tj4 = theor_just4_best(z_log_range)
mu_range_tj5 = theor_just5_best(z_log_range)

mu_range_opl2 = offpolylog2_best(z_log_range)
mu_range_opl3 = offpolylog3_best(z_log_range)
mu_range_opl4 = offpolylog4_best(z_log_range)
mu_range_opl5 = offpolylog5_best(z_log_range)

z_border = 1.4


def legend_IC(AIC, BIC):
    return '$\mathrm{AIC}=' + str(np.around(AIC, 1)) + ',\;\mathrm{BIC}=' + str(np.around(BIC, 1)) + '$'


fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(z_log_range, mu_range_pl1, zorder=5, c='gold')
ax.plot(z_log_range, mu_range_pl2, zorder=5)
ax.plot(z_log_range, mu_range_pl3, zorder=5)
ax.plot(z_log_range, mu_range_pl4, zorder=5)
ax.plot(z_log_range, mu_range_pl5, zorder=5)
# ax.axvline(z_border, c='k', linestyle='dashed')
ax.set_xlabel('redshift $z$')
ax.set_ylabel('distance modulus $\\mu$')
ax.set_title('Pantheon SN Ia Hubble diagram')
ax.legend(['Polylog., $p=1$, ' + legend_IC(AIC_polylog1, BIC_polylog1),
           'Polylog., $p=2$, ' + legend_IC(AIC_polylog2, BIC_polylog2),
           'Polylog., $p=3$, ' + legend_IC(AIC_polylog3, BIC_polylog3),
           'Polylog., $p=4$, ' + legend_IC(AIC_polylog4, BIC_polylog4),
           'Polylog., $p=5$, ' + legend_IC(AIC_polylog5, BIC_polylog5)], fancybox=True, shadow=True)
ax.errorbar(z_arr_SN, mu_arr_SN, Dmu_arr_SN, color='silver', linestyle='', marker='o', markeredgewidth=0.25,
            markeredgecolor='k', markersize=3.5, rasterized=True)
ax.set_xscale('log')
fig.tight_layout()
fig.savefig('./pics_misc/SNHD_polylog.pdf', dpi=225)

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.set_ylim((32, 48))
ax.plot(z_log_range, 5 * np.log10(c / H0 * z_log_range) + 25, zorder=5, c='black')
ax.plot(z_log_range, mu_range_tj2, zorder=5)
ax.plot(z_log_range, mu_range_tj3, zorder=5)
ax.plot(z_log_range, mu_range_tj4, zorder=5)
ax.plot(z_log_range, mu_range_tj5, zorder=5)
ax.axvline(z_border, c='k', linestyle='dashed')
ax.set_xlabel('redshift $z$')
ax.set_ylabel('distance modulus $\\mu$')
ax.set_title('Pantheon SN Ia Hubble diagram')
ax.legend(['Linear Hubble law', 'Theoretically justified, $p=2$, ' + legend_IC(AIC_theor_just2, BIC_theor_just2),
           'Theoretically justified, $p=3$, ' + legend_IC(AIC_theor_just3, BIC_theor_just3),
           'Theoretically justified, $p=4$, ' + legend_IC(AIC_theor_just4, BIC_theor_just4),
           'Theoretically justified, $p=5$, ' + legend_IC(AIC_theor_just5, BIC_theor_just5)], fancybox=True,
          shadow=True)
ax.errorbar(z_arr_SN, mu_arr_SN, Dmu_arr_SN, color='silver', linestyle='', marker='o', markeredgewidth=0.25,
            markeredgecolor='k', markersize=3.5, rasterized=True)
ax.set_xscale('log')
fig.tight_layout()
fig.savefig('./pics_misc/SNHD_theor_just.pdf', dpi=300)

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(z_log_range, mu_range_opl2, zorder=5)
ax.plot(z_log_range, mu_range_opl3, zorder=5)
ax.plot(z_log_range, mu_range_opl4, zorder=5)
ax.plot(z_log_range, mu_range_opl5, zorder=5)
ax.set_xlabel('redshift $z$')
ax.set_ylabel('distance modulus $\\mu$')
ax.set_title('Pantheon SN Ia Hubble diagram')
ax.legend(['Offset polylogarithmic, $p=2$, ' + legend_IC(AIC_offpolylog2, BIC_offpolylog2),
           'Offset polylogarithmic, $p=3$, ' + legend_IC(AIC_offpolylog3, BIC_offpolylog3),
           'Offset polylogarithmic, $p=4$, ' + legend_IC(AIC_offpolylog4, BIC_offpolylog4),
           'Offset polylogarithmic, $p=5$, ' + legend_IC(AIC_offpolylog5, BIC_offpolylog5)], fancybox=True, shadow=True)
ax.errorbar(z_arr_SN, mu_arr_SN, Dmu_arr_SN, color='silver', linestyle='', marker='o', markeredgewidth=0.25,
            markeredgecolor='k', markersize=3.5, rasterized=True)
ax.set_xscale('log')
fig.tight_layout()
fig.savefig('./pics_misc/SNHD_offpolylog.pdf', dpi=300)

# Самая лучшая модель -- полилогарифмическая с p=3. Создадим также Монте_Карло выборки для её параметров:

a1_sample, a2_sample = multivariate_normal.rvs(fit_tj2[0], fit_tj2[1], sample_size).T
fig = plt.figure(figsize=(5, 5))
fig = corner.corner(np.array([a1_sample, a2_sample]).T, fig=fig, labels=[r'$a_1$', r'$a_2$'],
                    quantiles=[sampling.ERR_QL, 0.5, sampling.ERR_QU], show_titles=True, bins=25)
fig.savefig('./pics/cornerplot_TJ2.pdf')


@jit(nopython=True)
def mu_SN_tj2_best(z):
    return polylog3(z, fit_tj2[0][0], fit_tj2[0][1], fit_tj2[0][2], fit_tj2[0][3])


@jit(nopython=True)
def de_SN_tj2(z, a1, a2):
    return (1 + z) ** (-0.5) * 10 ** ((theor_just2(z, a1, a2) - 25) / 5)


@jit(nopython=True)
def E_iso_SN_tj2(z, alpha, E_p, S_obs, a1, a2):
    return 4 * pi * (de_SN_tj2(z, a1, a2) * Mpc_in_cm) ** 2 * S_bolo(z, alpha, E_p, S_obs)


@jit(nopython=True)
def Amy_SN_tj2(z, alpha, E_p, S_obs, a1, a2):
    return np.log10(E_iso_SN_tj2(z, alpha, E_p, S_obs, a1, a2))


def sample_Amatixy_SN_tj2(ii, data):
    alpha_sample, E_p_sample, S_obs_sample, z_arg, a1_s, a2_s = data
    Amx_sample = Amx(z_arg[ii], E_p_sample)
    Amx_sample[np.isnan(Amx_sample)] = -np.inf
    Amy_sample = np.vectorize(lambda z, alpha, E_p, S_obs, a1, a2: Amy_SN_tj2(z, alpha, E_p, S_obs, a1, a2))(z_arg[ii],
                                                                                                             alpha_sample,
                                                                                                             E_p_sample,
                                                                                                             S_obs_sample,
                                                                                                             a1_s, a2_s)
    Amy_sample[np.isnan(Amy_sample)] = np.inf
    return (Amx_sample, Amy_sample)


# Находим значения x и y для плоскости Амати
print('Создание выборок для гамма-всплесков')


def loopfun_makesamples_SN_tj2(i):
    Amx_sample_SN_tj2, Amy_sample_SN_tj2 = sample_Amatixy_SN_tj2(i, (
        alpha_all_samples[i, :], E_p_all_samples[i, :], S_obs_all_samples[i, :], z_arr, a1_sample, a2_sample))
    return (Amx_sample_SN_tj2, Amy_sample_SN_tj2)


Amx_all_samples_SN_tj2, Amy_all_samples_SN_tj2 = np.array(list(
    zip(*Parallel(n_jobs=Nthreads, max_nbytes='2048M', verbose=10)(
        delayed(loopfun_makesamples_SN_tj2)(i) for i in np.arange(GRB_amount)[z_arr < 1.4]))))

# Для полученных выборок находим медианы, а также верхние и нижние пределы
Amx_meds_SN_tj2, Amx_dlim_SN_tj2, Amx_ulim_SN_tj2 = get_meds_and_lims(Amx_all_samples_SN_tj2)
Amy_meds_SN_tj2, Amy_dlim_SN_tj2, Amy_ulim_SN_tj2 = get_meds_and_lims(Amy_all_samples_SN_tj2)

# Спрямляем выборки
Amx_flat_SN_tj2 = Amx_all_samples_SN_tj2.flatten()
Amy_flat_SN_tj2 = Amy_all_samples_SN_tj2.flatten()

# Amx_lim_d = -0.75
# Amx_lim_u = 5.75
# Amy_lim_d = 40
# Amy_lim_u = 63.5
# limmask = (Amx_flat > Amx_lim_d) * (Amx_flat < Amx_lim_u) * (Amy_flat > Amy_lim_d) * (Amy_flat < Amy_lim_u)

# Для каждой реализации (их 1024 по умолчанию) находим коэффициенты a и b методом Тейла-Сена
a_sample_SN_tj2 = np.empty(sample_size)
b_sample_SN_tj2 = np.empty(sample_size)
for i in np.arange(sample_size):
    finitemask = np.isfinite(Amx_all_samples_SN_tj2[:, i]) * np.isfinite(Amy_all_samples_SN_tj2[:, i])
    fit_res = theilslopes(Amy_all_samples_SN_tj2[:, i][finitemask], Amx_all_samples_SN_tj2[:, i][finitemask])
    a_sample_SN_tj2[i] = fit_res[0]
    b_sample_SN_tj2[i] = fit_res[1]

# А здесь определяем среди этих выборок a и b медианы и погрешности
a_est_SN_tj2 = np.median(a_sample_SN_tj2)
b_est_SN_tj2 = np.median(b_sample_SN_tj2)
Da_est_SN_tj2 = (np.percentile(a_sample_SN_tj2, 100 * sampling.ERR_QU) - np.percentile(a_sample_SN_tj2,
                                                                                       100 * sampling.ERR_QL)) / 2
Db_est_SN_tj2 = (np.percentile(b_sample_SN_tj2, 100 * sampling.ERR_QU) - np.percentile(b_sample_SN_tj2,
                                                                                       100 * sampling.ERR_QL)) / 2

# Можем взглянуть на корреляцию между a и b:
fig = plt.figure(figsize=(5, 5))
fig = corner.corner(np.array([a_sample_SN_tj2, b_sample_SN_tj2]).T, fig=fig, labels=[r'$a$', r'$b$'],
                    quantiles=[sampling.ERR_QL, 0.5, sampling.ERR_QU], show_titles=True, bins=30)
fig.savefig('pics/cornerplot_ts_SN_tj2.pdf', dpi=225)

# fig = plt.figure(figsize=(5,5))
# fig = corner.corner(np.array([a_sample_SN_pl3, b_sample_SN_pl3]).T, fig=fig, labels=[r'$a$', r'$b$'], quantiles=[], show_titles=True, bins=30, color='Blue')
# corner.corner(np.array([a_sample, b_sample]).T, fig=fig, bins=30)
# fig.savefig('pics/cornerplot_ts_multiple.png', dpi=300)


# Далее нарисуем картинки:
finitemask = np.isfinite(Amx_flat_SN_tj2) * np.isfinite(Amy_flat_SN_tj2)
Amx_flat_SN_tj2 = Amx_flat_SN_tj2[finitemask]
Amy_flat_SN_tj2 = Amy_flat_SN_tj2[finitemask]

fig, ax = plt.subplots()
plot_Amati(fig, ax,
           (a_sample_SN_tj2, b_sample_SN_tj2, Amx_flat_SN_tj2, Amy_flat_SN_tj2, Amx_meds_SN_tj2, Amy_meds_SN_tj2),
           'Amati_ts_SN_tj2', 'repeated Theil-Sen estimation', plot_errbars=False, col2='darkblue')
fig, ax = plt.subplots()
plot_Amati(fig, ax, (
    a_sample_SN_tj2, b_sample_SN_tj2, Amx_meds_SN_tj2, Amy_meds_SN_tj2, Amx_ulim_SN_tj2, Amy_ulim_SN_tj2,
    Amx_dlim_SN_tj2,
    Amy_dlim_SN_tj2), 'Amati_ts_SN_tj2', 'repeated Theil-Sen estimation', plot_samples=False, col1='teal')
fig, ax = plt.subplots()
plot_Amati(fig, ax, (
    a_sample_SN_tj2, b_sample_SN_tj2, Amx_flat_SN_tj2, Amy_flat_SN_tj2, Amx_meds_SN_tj2, Amy_meds_SN_tj2,
    Amx_ulim_SN_tj2,
    Amy_ulim_SN_tj2, Amx_dlim_SN_tj2, Amy_dlim_SN_tj2), 'Amati_ts_SN_tj2', 'repeated Theil-Sen estimation')

# Зная a и b, найдём mu_A:
print('Расчёт mu_A для выборок гамма-всплесков с оценкой a и b методом Theil-Sen')


def loopfun_sample_mu_A_SN_tj2(i):
    return tuple(sample_mu_A(i, (alpha_all_samples[i, :], E_p_all_samples[i, :], S_obs_all_samples[i, :], z_arr),
                             a_sample_SN_tj2, b_sample_SN_tj2, 0.0))


mu_A_all_samples_SN_tj2 = np.array(list(zip(*Parallel(n_jobs=Nthreads, max_nbytes='2048M', verbose=10)(
    delayed(loopfun_sample_mu_A_SN_tj2)(i) for i in np.arange(GRB_amount))))).T

mu_A_meds_SN_tj2, mu_A_dlim_SN_tj2, mu_A_ulim_SN_tj2 = get_meds_and_lims(mu_A_all_samples_SN_tj2)
z_log_range = 10 ** np.linspace(np.min(np.log10(z_arr)), np.max(np.log10(z_arr)), 101)
mu_arr_st_SN_pl3 = np.empty(101)
for i in np.arange(101):
    mu_arr_st_SN_pl3[i] = mu(z_log_range[i], pars_cosm_planck70)

# Нарисуем теперь диаграмму Хаббла:
fig, ax = plt.subplots()
ax.plot(z_log_range, mu_arr_st, c='k', zorder=10)
ax.errorbar(z_arr, mu_A_meds_SN_tj2, yerr=np.array([mu_A_dlim_SN_tj2, mu_A_ulim_SN_tj2]), linestyle='', linewidth=0.3,
            marker='o', markersize=1.5, c='teal')
ax.set_xscale('log')
ax.set_ylim((32, 52))
ax.set_title('GRB Hubble Diagram')
ax.set_xlabel(r'$z$')
ax.set_ylabel(r'$\mu$')
fig.tight_layout()
fig.savefig('pics/HD_ts_SN_tj2.png', dpi=300)

z_log_range_big = 10 ** np.linspace(np.log10(np.min(z_arr_SN)), np.log10(np.max(z_arr)))
mu_range_lcdm70 = mu_SN_ufunc(z_log_range_big)
Delta_mu_arr_SN = mu_arr_SN - mu_SN_ufunc(z_arr_SN)
Delta_mu_A_meds_SN_tj2 = mu_A_meds_SN_tj2 - mu_SN_ufunc(z_arr)

fig, (ax, ax_low) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3.5, 1]}, figsize=(7, 6))
ax.errorbar(z_arr[z_arr < z_border], mu_A_meds_SN_tj2[z_arr < z_border],
            yerr=np.array([mu_A_dlim_SN_tj2[z_arr < z_border], mu_A_ulim_SN_tj2[z_arr < z_border]]), linestyle='',
            markeredgewidth=0.25, markeredgecolor='k', linewidth=0.3, marker='o', markersize=3, c='goldenrod', zorder=4,
            rasterized=True)
ax.errorbar(z_arr[z_arr >= z_border], mu_A_meds_SN_tj2[z_arr >= z_border],
            yerr=np.array([mu_A_dlim_SN_tj2[z_arr >= z_border], mu_A_ulim_SN_tj2[z_arr >= z_border]]), linestyle='',
            markeredgewidth=0.25, markeredgecolor='k', linewidth=0.3, marker='o', markersize=3, c='teal', zorder=3,
            rasterized=True)
ax.errorbar(z_arr_SN, mu_arr_SN, Dmu_arr_SN, color='silver', linestyle='', marker='o', markeredgewidth=0.25,
            markeredgecolor='k', markersize=2.5, rasterized=True)
ax.plot(z_log_range_big, mu_range_lcdm70, c='k', zorder=10)
ax.plot(z_log_range_big, 5 * np.log10(c / H0 * z_log_range_big) + 25, c='k', linewidth='1', linestyle='dashed',
        zorder=9)
ax_low.errorbar(z_arr[z_arr < z_border], Delta_mu_A_meds_SN_tj2[z_arr < z_border],
                yerr=np.array([mu_A_dlim_SN_tj2[z_arr < z_border], mu_A_ulim_SN_tj2[z_arr < z_border]]), linestyle='',
                markeredgewidth=0.25, markeredgecolor='k', linewidth=0.3, marker='o', markersize=3, c='goldenrod',
                zorder=4, rasterized=True)
ax_low.errorbar(z_arr[z_arr >= z_border], Delta_mu_A_meds_SN_tj2[z_arr >= z_border],
                yerr=np.array([mu_A_dlim_SN_tj2[z_arr >= z_border], mu_A_ulim_SN_tj2[z_arr >= z_border]]), linestyle='',
                markeredgewidth=0.25, markeredgecolor='k', linewidth=0.3, marker='o', markersize=3, c='teal', zorder=3,
                rasterized=True)
ax_low.errorbar(z_arr_SN, Delta_mu_arr_SN, Dmu_arr_SN, color='silver', linestyle='', marker='o', markeredgewidth=0.25,
                markeredgecolor='k', markersize=2.5, rasterized=True)
ax_low.plot(z_log_range_big, np.zeros(z_log_range_big.size), c='k', zorder=10)
ax_low.plot(z_log_range_big, - mu_range_lcdm70 + 5 * np.log10(c / H0 * z_log_range_big) + 25, c='k', linewidth='1',
            linestyle='dashed', zorder=9)
ax.legend(['$\Lambda$CDM', 'Linear Hubble Law', 'LGRBs with $z < 1.4$', 'LGRBs with $z \geq 1.4$', 'Pantheon SNe Ia'],
          fancybox=True, shadow=True)
ax.set_xscale('log')
ax_low.set_xscale('log')
ax.set_ylim((32, 52))
ax_low.set_ylim((-7.5, 7.5))
ax.set_title('SNe+LGRBs Hubble Diagram')
ax.set_ylabel(r'distance modulus $\mu$')
ax_low.set_ylabel(r'$\Delta\mu$')
ax_low.set_xlabel(r'redshift $z$')
fig.tight_layout()
fig.savefig('pics/HD_ts_SNGRB_tj2.png', dpi=225)
fig.savefig('pics/HD_ts_SNGRB_tj2.pdf', dpi=225)

fig, ([axlt, axrt], [axlb, axrb]) = plt.subplots(2, 2, sharex='col', sharey='row',
                                                 gridspec_kw={'height_ratios': [4, 1]}, figsize=(12.5, 7))
axlt.set_ylim((32, 52))
axlt.plot(z_log_range, 5 * np.log10(c / H0 * z_log_range) + 25, zorder=5, c='black')
axlt.plot(z_log_range, mu_range_tj2, zorder=6, c='tab:red')
axlt.plot(z_log_range, mu_range_tj3, zorder=5)
axlt.plot(z_log_range, mu_range_tj4, zorder=5)
axlt.plot(z_log_range, mu_range_tj5, zorder=5)
axlt.axvline(z_border, c='k', linestyle='dashed')
axlb.set_xlabel('redshift $z$')
axlt.set_ylabel('distance modulus $\\mu$')
axlt.set_title('Pantheon SN Ia Hubble diagram (log scale)')
axlt.legend(['Linear Hubble law', 'Theor. just., $p=2$, ' + legend_IC(AIC_theor_just2, BIC_theor_just2),
             'Theor. just., $p=3$, ' + legend_IC(AIC_theor_just3, BIC_theor_just3),
             'Theor. just., $p=4$, ' + legend_IC(AIC_theor_just4, BIC_theor_just4),
             'Theor. just., $p=5$, ' + legend_IC(AIC_theor_just5, BIC_theor_just5)], fancybox=True, shadow=True)
axlt.errorbar(z_arr_SN, mu_arr_SN, Dmu_arr_SN, color='silver', linestyle='', marker='o', markeredgewidth=0.25,
              markeredgecolor='k', markersize=3.5, rasterized=True)
axlt.set_xscale('log')

mu_tj2_SN_range = theor_just2_best(z_arr_SN)

axlb.set_ylim((-0.75, 0.75))
axlb.plot(z_log_range, 5 * np.log10(c / H0 * z_log_range) + 25, zorder=5, c='black')
axlb.plot(z_log_range, mu_range_tj2 - mu_range_tj2, zorder=6, c='tab:red')
axlb.plot(z_log_range, mu_range_tj3 - mu_range_tj2, zorder=5)
axlb.plot(z_log_range, mu_range_tj4 - mu_range_tj2, zorder=5)
axlb.plot(z_log_range, mu_range_tj5 - mu_range_tj2, zorder=5)
axlb.axvline(z_border, c='k', linestyle='dashed')
axlb.errorbar(z_arr_SN, mu_arr_SN - mu_tj2_SN_range, Dmu_arr_SN, color='silver', linestyle='', marker='o',
              markeredgewidth=0.25, markeredgecolor='k', markersize=3.5, rasterized=True)
axlb.set_ylabel('$\\Delta\\mu$')

mu_lcdm_range = np.vectorize(lambda z: mu(z, pars_cosm_planck70))(z_log_range)

axrt.plot(z_log_range, mu_lcdm_range, zorder=5, c='black')
axrt.plot(z_log_range, mu_range_tj2, zorder=5, c='tab:red')
axrt.axvline(z_border, c='k', linestyle='dashed')
axrb.set_xlabel('redshift $z$')
axrt.set_title('Pantheon SN Ia Hubble diagram (linear scale)')
axrt.legend(['$\Lambda$CDM model', 'Theor. just., $p=2$'], loc='upper left', fancybox=True, shadow=True)
axrt.errorbar(z_arr_SN, mu_arr_SN, Dmu_arr_SN, color='silver', linestyle='', marker='o', markeredgewidth=0.25,
              markeredgecolor='k', markersize=3.5, rasterized=True)
axrt.set_xscale('linear')

axrb.plot(z_log_range, mu_lcdm_range - mu_range_tj2, zorder=5, c='black')
axrb.plot(z_log_range, mu_range_tj2 - mu_range_tj2, zorder=5, c='tab:red')
axrb.axvline(z_border, c='k', linestyle='dashed')
axrb.errorbar(z_arr_SN, mu_arr_SN - mu_tj2_SN_range, Dmu_arr_SN, color='silver', linestyle='', marker='o',
              markeredgewidth=0.25, markeredgecolor='k', markersize=3.5, rasterized=True)

fig.tight_layout()
fig.savefig('./pics_misc/SNHD_theor_just_big.pdf')


# Для картинки со спектром:
@jit(nopython=True)
def E2NE(E, E_p, alpha):
    return E ** (2 + alpha) * np.exp(-(2 + alpha) * E / E_p) / (np.exp(-(2 + alpha) / E_p))


def E2NE_band(E_range, E_p_sample, alpha_sample, conf_int=100 * (sampling.ERR_QU - sampling.ERR_QL)):
    upper_bord = np.vectorize(lambda E: np.percentile(
        E ** (2 + alpha_sample) * np.exp(-(2 + alpha_sample) * E / E_p_sample) / (
            np.exp(-(2 + np.median(alpha_sample)) / np.median(E_p_sample))), (100 + conf_int) / 2))(E_range)
    lower_bord = np.vectorize(lambda E: np.percentile(
        E ** (2 + alpha_sample) * np.exp(-(2 + alpha_sample) * E / E_p_sample) / (
            np.exp(-(2 + np.median(alpha_sample)) / np.median(E_p_sample))), (100 - conf_int) / 2))(E_range)
    return (lower_bord, upper_bord)


E_range_specplot = 10 ** np.linspace(0, 3, 100)
E2NE_range_specplot = E2NE(E_range_specplot, E_p_arr[100], alpha_arr[100])
E2NE_range_1sig_d, E2NE_range_1sig_u = E2NE_band(E_range_specplot, E_p_all_samples[100, :], alpha_all_samples[100, :])

fig, ax = plt.subplots(figsize=(5.5, 4))
ax.grid(linestyle='dashed')
ax.plot(E_range_specplot, E2NE_range_specplot, zorder=20)
ax.fill_between(E_range_specplot, E2NE_range_1sig_d, E2NE_range_1sig_u, alpha=0.2, color='k', linestyle=':',
                linewidth=0.75)
ax.axvline(E_p_arr[100], c='red')
ax.axvline(E_p_arr[100] / (2 + alpha_arr[100]), c='k', linestyle='dashed')
ax.set_ylim((10 ** -1.6, 100))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Energy $E,\\,\\mathrm{keV}$')
ax.set_ylabel('$E^2N(E)$')
ax.legend(['spectrum', '$E_\\mathrm{p}$', '$E_\\mathrm{p} / (2 + \\alpha)$', '$1\\sigma$-conf. int. for spectrum'],
          labelspacing=0.15, fancybox=True, shadow=True)
ax.set_title('The GRB' + GRB_names[100] + ' spectrum at $z = {:.5}$'.format(
    z_arr[100]) + ',\n' + '$E_\\mathrm{p}' + ' = {:.3}'.format(E_p_arr[100]) + '^{+' + '{:.2}'.format(
    DE_p_arr_u[100]) + '}_{-' + '{:.2}'.format(DE_p_arr_d[100]) + '}$, ' + '$\\alpha = {:.3}0'.format(
    alpha_arr[100]) + '\\pm{:.2}'.format(Dalpha_arr_u[100]) + '}$')
fig.tight_layout()
fig.savefig('./pics_misc/CPL_spectra_ex.pdf')


# Определим космологические модели

def mu_2par_residuals(pars, z_arg, mu_A_arg):
    return np.vectorize(lambda z: mu(z, (70, pars[0], pars[1], pars[0] + pars[1] - 1)))(z_arg) - mu_A_arg


def mu_1par_residuals(pars, z_arg, mu_A_arg):
    return np.vectorize(lambda z: mu(z, (70, pars[0], 1 - pars[0], 0.0)))(z_arg) - mu_A_arg


def loopfun_lcdm(i):
    Omm_2p, OmDE_2p = least_squares(mu_2par_residuals, (0.3, 0.7), loss='soft_l1', f_scale=1.0,
                                    bounds=(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
                                    args=(z_arr, mu_A_inv_all_samples[:, i])).x
    Omm_1p = \
        least_squares(mu_1par_residuals, (0.3), loss='soft_l1', f_scale=1.0,
                      args=(z_arr, mu_A_inv_all_samples[:, i])).x[0]
    return (Omm_2p, OmDE_2p, Omm_1p)


Omm_2p_sample, OmDE_2p_sample, Omm_1p_sample = np.array(list(
    zip(*Parallel(n_jobs=Nthreads, max_nbytes='2048M', verbose=10)(
        delayed(loopfun_lcdm)(i) for i in np.arange(sample_size)))))

fig, ax = plt.subplots()
ax.hist(Omm_1p_sample, bins=40)


def getlegend(pars_cosm):
    H0, Omm, OmDE, Omk = pars_cosm
    return r'$H_0 = ' + str(
        np.around(H0, decimals=1)) + r'\,\frac{\mathrm{km/s}}{\mathrm{Mpc}},\; \Omega_{\mathrm{m}} = ' + str(
        np.around(Omm, decimals=3)) + r',\; \Omega_{\mathrm{DE}} = ' + str(
        np.around(OmDE, decimals=3)) + r',\; \Omega_{\mathrm{k}} = ' + str(np.around(Omk, decimals=3)) + r'$'


Omm_2p_med = np.median(Omm_2p_sample)
OmDE_2p_med = np.median(OmDE_2p_sample)
Omm_1p_med = np.median(Omm_1p_sample)

mu_arr_2p = np.vectorize(lambda z: mu(z, (70, Omm_2p_med, OmDE_2p_med, Omm_2p_med + OmDE_2p_med - 1)))(z_log_range)
mu_arr_1p = np.vectorize(lambda z: mu(z, (70, Omm_1p_med, 1 - Omm_1p_med, 0.0)))(z_log_range)

fig, ax = plt.subplots()
ax.plot(z_log_range, mu_arr_st, c='k', zorder=10)
# ax.plot(z_log_range, mu_arr_2p, c='tab:orange', zorder=8)
ax.plot(z_log_range, mu_arr_1p, c='green', zorder=11)
ax.errorbar(z_arr, mu_A_inv_meds, yerr=np.array([mu_A_inv_dlim, mu_A_inv_ulim]), linestyle='', linewidth=0.3,
            marker='o', markersize=1.5, c='teal')
ax.set_xscale('log')
ax.set_ylim((32, 52))
ax.set_title('GRB Hubble Diagram')
ax.set_xlabel(r'$z$')
ax.set_ylabel(r'$\mu$')
fig.tight_layout()
# fig.savefig('pics/HD_icf.png', dpi=300)

qst = np.linspace(0, 1, 1002)[1:-1]
xst = np.vectorize(sampling.smooth_split_normal_ppf)(qst, 5, 1.0, 2.0)
xst2 = np.linspace(xst[0], xst[-1], 1000)
qst2 = np.vectorize(sampling.split_normal_cdf)(xst2, 5, 1.0, 2.0)
qst3 = np.vectorize(sampling.offset_lognorm_cdf)(xst2, 5, 1.0, 2.0)
fig, ax = plt.subplots()
ax.plot(xst, qst)
ax.plot(xst2, qst2)
ax.plot(xst2, qst3)
ax.plot(np.array([4, 5, 7]), np.array([sampling.ERR_QL, 0.5, sampling.ERR_QU]), linestyle='', marker='o', c='k',
        markersize=2)

yst2 = 1 / np.gradient(xst, qst)
yst3 = np.gradient(qst2, xst2)
yst4 = np.gradient(qst3, xst2)
fig, ax = plt.subplots()
ax.plot(xst2, yst3, linestyle='dashed', color='darkgray')
# ax.plot(xst2, yst4, color='darkgray')
ax.plot(xst, yst2, color='k')

test = sampling.random_smooth_split_normal(5, 2, 1, 10000)
fig, ax = plt.subplots()
ax.hist(test, bins=50)

xst2 = np.linspace(0.16, 0.84, 51)
yst20 = (np.cos((xst2 - 0.5) / (sampling.ERR_QU - 0.5) * pi) + 1) / 2.0
yst21 = np.sqrt(1 - ((xst2 - 0.5) / (sampling.ERR_QU - 0.5)) ** 2)
fig, ax = plt.subplots()
ax.plot(xst2, yst20)
ax.plot(xst2, yst21)
