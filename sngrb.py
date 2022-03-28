import numpy as np
from scipy.optimize import least_squares, curve_fit, fmin
from scipy.stats.mstats import theilslopes
from scipy.stats import skewnorm, norm, multivariate_normal
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rc, rcParams
import corner
from joblib import Parallel, delayed, cpu_count
#from uncertainties import wrap as uw
from numba import jit, vectorize
import warnings
warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore', over='ignore')

#Назначаем константы:
pi = 4 * np.arctan(1.0)
c = 299792.458 #В километрах в секунду
H0 = 70
pars_cosm_0 = (70, 0.3, 0.7, 0.0)
pars_cosm_planck = (67.74, 0.3089, 0.6911, 0.0)
pars_cosm_planck70 = (70, 0.3089, 0.6911, 0.0)
pars_cosm_SN = (70, 0.309347240700862, 0.690652759299138, 0.0)
Mpc_in_cm = 3.08567758e24

err_ql = norm.cdf(-1.0)
err_qu = norm.cdf(1.0)

paper_linewidth = 3.37689 #Ширина колонки текста в MNRAS-овском шаблоне
paper_textwidth = 7.03058 #Ширина страницы в MNRAS-овском шаблоне

#Натройка шрифтов в matplotlib:
rc('font', family = 'Times New Roman')
rc('font', size=10)
rc('text', usetex=False)
rc('text.latex',preamble=r'\usepackage[utf8]{inputenc}')
rc('text.latex',preamble=r'\usepackage[russian]{babel}')
rc('mathtext', fontset='stix')
rc('figure', figsize = (5, 4.25))

#Оно же, но для Universe:
#rc('font', family = 'Palatino Linotype')
rc('font', family = 'tex gyre pagella')
rcParams['font.sans-serif'] = 'tex gyre pagella'
rc('font', size=12)
rc('text', usetex=False)
rc('text.latex',preamble=r'\usepackage[utf8]{inputenc}')
rc('text.latex',preamble=r'\usepackage[russian]{babel}')
#rc('text.latex',preamble=r'\usepackage{newpxtext}')
#rc('text.latex',preamble=r'\usepackage{newpxmath}')
rc('mathtext', fontset='custom', it='tex gyre pagella:italic')
rc('figure', figsize = (5, 4.25))

#Важный параметр -- размер выборок Монте-Карло:
#Для тестирования можно брать значение 1024 -- с ним код прогоняется недолго
#Для окончательной прогонки лучше брать значения побольше -- например, 10000
sample_size = 10000

#Количество параллельных процессов:
Nthreads = cpu_count()

#==============================================================================
#    Далее идёт часть кода, в которой я определяю различные нужные функции
#==============================================================================

#Обычная линейная функция (нужна для фиттинга)
def lin(x, A, B):
    return A * x + B

#Функция, возвращающая верхнюю и нижнюю 1-сигма ограничивающую кривую для линейной функции
def lin_band(xs, a_sample, b_sample, conf_int=100 * (err_qu - err_ql)):
    upper_bord = np.vectorize(lambda x: np.percentile(lin(x, a_sample, b_sample), (100 + conf_int) / 2))(xs)
    lower_bord = np.vectorize(lambda x: np.percentile(lin(x, a_sample, b_sample), (100 - conf_int) / 2))(xs)
    return (lower_bord, upper_bord)

#Тоже линейная функция, но с переставленными x и y
def lin_inv(y, A, B):
    return (y - B) / A

def split_normal_cdf(x: float, mu: float, lower_sigma: float, upper_sigma: float):
    if x < mu:
        return norm.cdf(x, mu, lower_sigma)
    else:
        return norm.cdf(x, mu, upper_sigma)

#Создание выборки из разделённо-нормального распределения
def random_split_normal(mu: float, lower_sigma: float, upper_sigma: float, amount: int = 1024):
    z = np.random.normal(0, 1, amount)
    return mu + z * (- lower_sigma * (np.sign(z) - 1) / 2 + upper_sigma * (np.sign(z) + 1) / 2)

#Создание выборки из логарифмического разделённо-нормального распределения
def random_log_split_normal(mu: float, lower_sigma: float, upper_sigma: float, amount: int = 1024):
    return 10 ** (random_split_normal(np.log10(mu), np.log10(mu / (mu - lower_sigma)), np.log10((mu + upper_sigma) / mu), amount = amount))

@jit
def twopiece_normal_ppf(q: float, mu: float, lower_sigma: float, upper_sigma: float):
    if q < lower_sigma / (lower_sigma + upper_sigma):
        return norm.ppf(q * (lower_sigma + upper_sigma) / 2.0 / lower_sigma, mu, lower_sigma)
    else:
        return norm.ppf((q * (lower_sigma + upper_sigma) - lower_sigma + upper_sigma) / 2.0 / upper_sigma, mu, upper_sigma)

@vectorize(['float64(float64,float64,float64,float64)'])
def twopiece_normal_ppf_ufunc(q, mu, lsigma, usigma):
    return twopiece_normal_ppf(q, mu, lsigma, usigma)

def random_twopiece_normal(mu: float, lower_sigma: float, upper_sigma: float, amount: int = 1024):
    return np.vectorize(twopiece_normal_ppf)(np.random.uniform(size=amount), mu, lower_sigma, upper_sigma)

def random_offset_lognormal(mu: float, lower_sigma: float, upper_sigma: float, amount: int = 1024):
    if lower_sigma == upper_sigma:
        return np.random.normal(mu, upper_sigma, size=amount)
    elif lower_sigma < upper_sigma:
        delta = mu - upper_sigma * lower_sigma / (upper_sigma - lower_sigma)
        return 10 ** np.random.normal(np.log10(mu - delta), np.log10((mu + upper_sigma - delta) / (mu - delta)), size=amount) + delta
    else:
        delta = mu - lower_sigma * upper_sigma / (lower_sigma - upper_sigma)
        return - 10 ** np.random.normal(np.log10(mu - delta), np.log10((mu + lower_sigma - delta) / (mu - delta)), size=amount) - delta + 2 * mu

def offset_lognorm_cdf(x, mu, lower_sigma, upper_sigma):
    if lower_sigma < upper_sigma:
        delta = mu - lower_sigma * upper_sigma / (upper_sigma - lower_sigma)
        return norm.cdf(np.log10(x-delta), np.log10(mu-delta), np.log10((mu+upper_sigma-delta) / (mu-delta)))
    elif lower_sigma > upper_sigma:
        delta = mu - lower_sigma * upper_sigma / (lower_sigma - upper_sigma)
        return 1 - norm.cdf(np.log10(-x+2*mu-delta), np.log10(mu-delta), np.log10((mu+lower_sigma-delta) / (mu-delta)))
    else:
        return norm.cdf(x, mu, upper_sigma)

def offset_lognorm_ppf(q, mu, lower_sigma, upper_sigma):
    if lower_sigma < upper_sigma:
        delta = mu - lower_sigma * upper_sigma / (upper_sigma - lower_sigma)
        return 10 ** norm.ppf(q, np.log10(mu-delta), np.log10((mu+upper_sigma-delta) / (mu-delta))) + delta
    elif lower_sigma > upper_sigma:
        delta = mu - lower_sigma * upper_sigma / (lower_sigma - upper_sigma)
        return - 10 ** norm.ppf(1-q, np.log10(mu-delta), np.log10((mu+lower_sigma-delta) / (mu-delta))) - delta + 2 * mu
    else:
        return norm.ppf(q, mu, upper_sigma)

def offset_lognorm_pdf(x, mu, lower_sigma, upper_sigma):
    if lower_sigma < upper_sigma:
        delta = mu - lower_sigma * upper_sigma / (upper_sigma - lower_sigma)
        return norm.pdf(np.log10(x-delta), np.log10(mu-delta), np.log10((mu+upper_sigma-delta) / (mu-delta))) / (x - delta) / np.log(10)
    elif lower_sigma > upper_sigma:
        delta = mu - lower_sigma * upper_sigma / (lower_sigma - upper_sigma)
        return norm.pdf(np.log10(-x+2*mu-delta), np.log10(mu-delta), np.log10((mu+lower_sigma-delta) / (mu-delta))) / (2 * mu - x - delta) / np.log(10)
    else:
        return norm.pdf(x, mu, upper_sigma)

#def smooth_split_normal_ppf(q, mu, lower_sigma, upper_sigma):
#    if (q > err_ql) and (q < err_qu):
#        w = (np.cos((q - 0.5) / (err_qu - 0.5) * pi) + 1) / 2.0
#        #w = 1 - np.abs((q - 0.5) / (err_qu - 0.5))
#        if q < 0.5:
#            return offset_lognorm_ppf(q, mu, lower_sigma, upper_sigma) * w + norm.ppf(q, mu, lower_sigma) * (1 - w)
#        else:
#            return offset_lognorm_ppf(q, mu, lower_sigma, upper_sigma) * w + norm.ppf(q, mu, upper_sigma) * (1 - w)
#    elif q <= err_ql:
#        return norm.ppf(q, mu, lower_sigma)
#    else:
#        return norm.ppf(q, mu, upper_sigma)

def smooth_split_normal_ppf(q, mu, lower_sigma, upper_sigma):
    if (q > norm.cdf(-2.0)) and (q < 0.5):
        x = (q - 0.5) / (0.5 - norm.cdf(-2.0))
        w = (np.cos(x * pi) + 1) / 2.0
        return offset_lognorm_ppf(q, mu, lower_sigma, upper_sigma) * w + norm.ppf(q, mu, lower_sigma) * (1 - w)
    elif (q >= 0.5) and (q < norm.cdf(2.0)):
        x = (q - 0.5) / (norm.cdf(2.0) - 0.5)
        w = (np.cos(x * pi) + 1) / 2.0
        return offset_lognorm_ppf(q, mu, lower_sigma, upper_sigma) * w + norm.ppf(q, mu, upper_sigma) * (1 - w)
    elif q <= norm.cdf(-2.0):
        return norm.ppf(q, mu, lower_sigma)
    else:
        return norm.ppf(q, mu, upper_sigma)

def random_smooth_split_normal(mu, lower_sigma, upper_sigma, amount=1024):
    return np.vectorize(smooth_split_normal_ppf)(np.random.uniform(size=amount), mu, lower_sigma, upper_sigma)

#Задание констант: число промежутков интегрирования
intit_dist = 10001
intit_sp = 10001
intit_sperr = 10001

#Далее идёт задание космологических функций с jit-компиляцией
@jit(nopython=True)
def r(z, pars_cosm, am=intit_dist):
    H0, Omm, OmDE, Omk = pars_cosm
    z_int_range = np.linspace(0, z, am)
    h_int_range = 1 / h(z_int_range, pars_cosm)
    z_int = np.trapz(h_int_range, z_int_range)
    return c / H0 * z_int

@jit(nopython=True)
def h(z, pars_cosm):
    H0, Omm, OmDE, Omk = pars_cosm
    return np.sqrt(Omm * (1 + z) ** 3 + OmDE - Omk * (1 + z) ** 2)

@jit(nopython=True)
def l(z, pars_cosm, am=intit_dist):
    H0, Omm, OmDE, Omk = pars_cosm
    z_int_range = np.linspace(0, z, am)
    h_int_range = 1 / h(z_int_range, pars_cosm)
    z_int = np.trapz(h_int_range, z_int_range)
    if Omk == 0:
        return c / H0 * z_int
    elif Omk > 0:
        return c / H0 / np.sqrt(np.abs(Omk)) * np.sin(np.sqrt(np.abs(Omk)) * z_int)
    else:
        return c / H0 / np.sqrt(np.abs(Omk)) * np.sinh(np.sqrt(np.abs(Omk)) * z_int)
    
@jit(nopython=True)
def dl(z, pars_cosm):
    H0, Omm, OmDE, Omk = pars_cosm
    return (1 + z) * l(z, pars_cosm)

@jit(nopython=True)
def de(z, pars_cosm):
    H0, Omm, OmDE, Omk = pars_cosm
    return np.sqrt(1 + z) * l(z, pars_cosm)

@jit(nopython=True)
def mu(z, pars_cosm):
    H0, Omm, OmDE, Omk = pars_cosm
    return 25 + 5 * np.log10(dl(z, pars_cosm))

@vectorize(['float64(float64)'])
def mu_SN_ufunc(z):
    return mu(z, pars_cosm_planck70)

@jit(nopython=True)
def EN(E, pars_spec):
    alpha, E_p = pars_spec
    E_0 = E_p / (2 + alpha)
    return E ** (alpha + 1) * np.exp(-E / E_0)

@jit(nopython=True)
def int_E(lims, pars_spec, am=intit_sp):
    E_range = np.linspace(lims[0], lims[1], am)
    EN_range = EN(E_range, pars_spec)
    return np.trapz(EN_range, E_range)

@jit(nopython=True)
def S_bolo(z, alpha, E_p, S_obs):
    denom_int = int_E((15, 150), (alpha, E_p))
    num_int = int_E((1 / (1 + z), 1e4 / (1 + z)), (alpha, E_p))
    if denom_int == 0:
        return np.inf
    else:
        return S_obs * num_int / denom_int * 1e-7

@jit(nopython=True)
def E_iso(z, alpha, E_p, S_obs, pars_cosm):
    return 4 * pi * (de(z, pars_cosm) * Mpc_in_cm) ** 2 * S_bolo(z, alpha, E_p, S_obs)

@jit(nopython=True)
def mu_A(z, alpha, E_p, S_obs, a, b, k_par):
    S_bolo_arg = S_bolo(z, alpha, E_p, S_obs)
    if S_bolo_arg == 0 or S_bolo_arg > 1e10:
        return 0
    else:
        return 25 + 2.5 * (np.log10((1 + z) ** (k_par + 1) / (4 * pi * S_bolo_arg * Mpc_in_cm ** 2)) + a * np.log10(E_p * (1 + z)) + b)

@vectorize(['float64(float64,float64,float64,float64,float64,float64,float64)'])
def mu_A_ufunc(z, alpha, E_p, S_obs, a, b, k_par):
    return mu_A(z, alpha, E_p, S_obs, a, b, k_par)

@jit(nopython=True)
def Amx(z, E_p):
    return np.log10(E_p * (1 + z))

@jit(nopython=True)
def Amy(z, alpha, E_p, S_obs, pars_cosm):
    return np.log10(E_iso(z, alpha, E_p, S_obs, pars_cosm))

#uE_iso = uw(E_iso)
#umu_A = uw(mu_A)
#uAmx = uw(Amx)
#uAmy = uw(Amy)

#Функция, создающая выборки для alpha, Ep и Sobs
def make_samples_split(ii, data, amount=1024):
    alpha_arg, Dalpha_arg_d, Dalpha_arg_u, E_p_arg, DE_p_arg_d, DE_p_arg_u, S_obs_arg, DS_obs_arg, z_arg = data
    alpha_sample = random_split_normal(alpha_arg[ii], Dalpha_arg_d[ii], Dalpha_arg_u[ii], amount)
    E_p_sample = random_log_split_normal(E_p_arg[ii], DE_p_arg_d[ii], DE_p_arg_u[ii], amount=amount)
    #E_p_sample[E_p_sample <= 1e-3] = 1e-3
    S_obs_sample = np.random.normal(S_obs_arg[ii], DS_obs_arg[ii], amount)
    return (alpha_sample, E_p_sample, S_obs_sample)

#Она же, но со скошенным нормальным распределением -- результаты хуже
def make_samples_skew(ii, data, amount=1024):
    alpha_arg, Dalpha_arg_d, Dalpha_arg_u, E_p_arg, DE_p_arg_d, DE_p_arg_u, S_obs_arg, DS_obs_arg, z_arg = data
    fit_bounds = (np.array([-50, -np.inf, -np.inf]), np.array([50, np.inf, np.inf]))
    alpha_fit = curve_fit(skewnorm.cdf, np.array([alpha_arr_d[ii], alpha_arr[ii], alpha_arr_u[ii]]), np.array([0.16, 0.5, 0.84]), p0=np.array([0.0, alpha_arr[ii], (alpha_arr_u[ii] - alpha_arr_d[ii]) / 2]), bounds=fit_bounds)
    alpha_sample = skewnorm.rvs(alpha_fit[0][0], alpha_fit[0][1], alpha_fit[0][2], amount)
    E_p_fit = curve_fit(skewnorm.cdf, np.log10(np.array([E_p_arr_d[ii], E_p_arr[ii], E_p_arr_u[ii]])), np.array([0.16, 0.5, 0.84]), p0=np.array([0.0, np.log10(E_p_arr[ii]), 0.5 * np.log10(E_p_arr_u[ii] / E_p_arr_d[ii])]), bounds=fit_bounds)
    E_p_sample = 10 ** skewnorm.rvs(E_p_fit[0][0], E_p_fit[0][1], E_p_fit[0][2], amount)
    S_obs_sample = np.random.normal(S_obs_arg[ii], DS_obs_arg[ii], amount)
    return (alpha_sample, E_p_sample, S_obs_sample)

#Смещённое логарифмическое распределение:
def make_samples(ii, data, amount=1024):
    alpha_arg, Dalpha_arg_d, Dalpha_arg_u, E_p_arg, DE_p_arg_d, DE_p_arg_u, S_obs_arg, DS_obs_arg, z_arg = data
    alpha_sample = random_smooth_split_normal(alpha_arg[ii], Dalpha_arg_d[ii], Dalpha_arg_u[ii], amount)
    #E_p_sample = random_smooth_split_normal(E_p_arg[ii], DE_p_arg_d[ii], DE_p_arg_u[ii], amount=amount)
    E_p_sample = 10 ** (random_smooth_split_normal(np.log10(E_p_arg[ii]), np.log10(E_p_arg[ii] / (E_p_arg[ii] - DE_p_arg_d[ii])), np.log10((E_p_arg[ii] + DE_p_arg_u[ii]) / E_p_arg[ii]), amount = amount))
    S_obs_sample = np.random.normal(S_obs_arg[ii], DS_obs_arg[ii], amount)
    return (alpha_sample, E_p_sample, S_obs_sample)

def make_samples_twopiece(ii, data, amount=1024):
    alpha_arg, Dalpha_arg_d, Dalpha_arg_u, E_p_arg, DE_p_arg_d, DE_p_arg_u, S_obs_arg, DS_obs_arg, z_arg = data
    alpha_sample = random_twopiece_normal(alpha_arg[ii], Dalpha_arg_d[ii], Dalpha_arg_u[ii], amount)
    #E_p_sample = random_twopiece_normal(E_p_arg[ii], DE_p_arg_d[ii], DE_p_arg_u[ii], amount=amount)
    E_p_sample = 10 ** (random_twopiece_normal(np.log10(E_p_arg[ii]), np.log10(E_p_arg[ii] / (E_p_arg[ii] - DE_p_arg_d[ii])), np.log10((E_p_arg[ii] + DE_p_arg_u[ii]) / E_p_arg[ii]), amount = amount))
    S_obs_sample = np.random.normal(S_obs_arg[ii], DS_obs_arg[ii], amount)
    return (alpha_sample, E_p_sample, S_obs_sample)

#Функция, возвращающая медианы и пределы доверительного интервала для выборок
def get_meds_and_lims(samples):
    N_inner = np.shape(samples)[0]
    meds = np.empty(N_inner)
    dlim = np.empty(N_inner)
    ulim = np.empty(N_inner)
    for i in np.arange(N_inner):
        meds[i] = np.median(samples[i,:][np.isfinite(samples[i,:])])
        dlim[i] = meds[i] - np.percentile(samples[i,:][np.isfinite(samples[i,:])], 100 * err_ql)
        ulim[i] = np.percentile(samples[i,:][np.isfinite(samples[i,:])], 100 * err_qu) - meds[i]
    return (meds, dlim, ulim)

def twopiece_lhood(mu, data):
    data_arg = data[np.isfinite(data)]
    return - np.sum((data_arg[data_arg <= mu] - mu) ** 2) ** (1 / 3) - np.sum((data_arg[data_arg > mu] - mu) ** 2) ** (1 / 3)

def get_twopiece_pars(samples):
    if np.size(np.shape(samples)) == 1:
        mu_est = fmin(lambda mu_arg: - twopiece_lhood(mu_arg, samples), 0.0, disp=False)[0]
        sample_cl = samples[np.isfinite(samples)]
        L_est = twopiece_lhood(mu_est, sample_cl)
        s1_est = np.sqrt(- L_est / np.size(sample_cl) * (np.sum((sample_cl[sample_cl <= mu_est] - mu_est) ** 2)) ** (2 / 3))
        s2_est = np.sqrt(- L_est / np.size(sample_cl) * (np.sum((sample_cl[sample_cl > mu_est] - mu_est) ** 2)) ** (2 / 3))
        return (mu_est, s1_est, s2_est)
    else:
        N_inner = np.shape(samples)[0]
        mu_ests = np.empty(N_inner)
        s1_ests = np.empty(N_inner)
        s2_ests = np.empty(N_inner)
        for i in np.arange(N_inner):
            mu_ests[i] = fmin(lambda mu_arg: -twopiece_lhood(mu_arg, samples[i,:]), 0.0, disp=False)[0]
            sample_cl = samples[i,:][np.isfinite(samples[i,:])]
            L_est = twopiece_lhood(mu_ests[i], sample_cl)
            s1_ests[i] = np.sqrt(- L_est / np.size(sample_cl) * (np.sum((sample_cl[sample_cl <= mu_ests[i]] - mu_ests[i]) ** 2)) ** (2 / 3))
            s2_ests[i] = np.sqrt(- L_est / np.size(sample_cl) * (np.sum((sample_cl[sample_cl > mu_ests[i]] - mu_ests[i]) ** 2)) ** (2 / 3))
        return (mu_ests, s1_ests, s2_ests)

#Функция, вычисляющая x и y в плоскости Амати для выборок
def sample_Amatixy(ii, data):
    alpha_sample, E_p_sample, S_obs_sample, z_arg = data
    Amx_sample = Amx(z_arg[ii], E_p_sample)
    Amx_sample[np.isnan(Amx_sample)] = -np.inf
    Amy_sample = np.vectorize(lambda z, alpha, E_p, S_obs: Amy(z, alpha, E_p, S_obs, pars_cosm_planck70))(z_arg[ii], alpha_sample, E_p_sample, S_obs_sample)
    Amy_sample[np.isnan(Amy_sample)] = np.inf
    return (Amx_sample, Amy_sample)

#Функция, вычисляющая Sbolo для выборок
def sample_S_bolo(ii, data):
    alpha_sample, E_p_sample, S_obs_sample, z_arg = data
    S_bolo_sample = S_bolo(z_arg[ii], alpha_sample, E_p_sample, S_obs_sample) * Mpc_in_cm ** 2
    return S_bolo_sample

#Функция, вычисляющая mu_A для выборок
def sample_mu_A(ii, data, a, b, k):
    alpha_sample, E_p_sample, S_obs_sample, z_arg = data
    mu_A_sample = np.vectorize(mu_A)(z_arg[ii], alpha_sample, E_p_sample, S_obs_sample, a, b, k)
    return mu_A_sample

#Отрисовка красивых графиков в плоскости Амати
def plot_Amati(fig, ax, data, imagename, legendname, plot_errbars=True, plot_samples=True, xlim = (0.75, 4.25), ylim = (49, 55), col1='darkgreen', col2='darkgreen', cmap='Greens'):
    if plot_errbars and plot_samples:
        a_sample, b_sample, Amx_flat, Amy_flat, Amx_meds, Amy_meds, Amx_ulim, Amy_ulim, Amx_dlim, Amy_dlim = data
    elif plot_errbars:
        a_sample, b_sample, Amx_meds, Amy_meds, Amx_ulim, Amy_ulim, Amx_dlim, Amy_dlim = data
    elif plot_samples:
        a_sample, b_sample, Amx_flat, Amy_flat, Amx_meds, Amy_meds = data
    a_est = np.median(a_sample)
    b_est = np.median(b_sample)
    Da_est = (np.percentile(a_sample, 100 * err_qu) - np.percentile(a_sample, 100 * err_ql)) / 2
    Db_est = (np.percentile(b_sample, 100 * err_qu) - np.percentile(b_sample, 100 * err_ql)) / 2
    x_range = np.linspace(xlim[0], xlim[1], 100)
    y_range = lin(x_range, a_est, b_est)
    y_range_1sig_d, y_range_1sig_u = lin_band(x_range, a_sample, b_sample)
    Am_res_sigma = (Amy_meds - np.median(a_sample) * Amx_meds - np.median(b_sample)).std()
    ax.plot(x_range, y_range, c='k', zorder=10)
    ax.plot([], [], c='grey', linestyle=':', linewidth=0.75)
    ax.plot(x_range, y_range + Am_res_sigma, c='k', zorder=10, linestyle='dashed', linewidth=1)
    ax.fill_between(x_range, y_range_1sig_d, y_range_1sig_u, alpha=0.2, color='k', linestyle=':', linewidth=0.75)
    ax.plot(x_range, y_range - Am_res_sigma, c='k', zorder=10, linestyle='dashed', linewidth=1)
    if plot_errbars:
        ax.errorbar(Amx_meds, Amy_meds, yerr=np.array([Amy_dlim, Amy_ulim]), xerr=np.array([Amx_dlim, Amx_ulim]), linestyle='', linewidth=0.3, marker='o', markersize=1.15, color=col1, rasterized=True)
    if plot_samples:
        limmask = (Amx_flat > xlim[0]) * (Amx_flat < xlim[1]) * (Amy_flat > ylim[0]) * (Amy_flat < ylim[1])
        ax.scatter(Amx_flat[limmask], Amy_flat[limmask], s=0.0015 * (1000 / sample_size) ** 1.5, c=col2, marker=".", rasterized=True)
        #ax.scatter_density(Amx_flat[limmask], Amy_flat[limmask], color=col2, vmin=0, vmax=50)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_title('Amati plane, ' + legendname)
    ax.set_xlabel(r'$\mathrm{log}\,(E_{\mathrm{p,i}} \,/\, 1\,\mathrm{keV})$')
    ax.set_ylabel(r'$\mathrm{log}\,(E_{\mathrm{iso}}\,/\, 1\,\mathrm{erg})$')
    ax.legend([ '$a=' + str(np.around(a_est, 2)) + '\\pm' + str(np.around(Da_est, 2)) + '$,\n$b=' + str(np.around(b_est, 1)) + '\\pm' + str(np.around(Db_est, 1)) + '$', '$1\\sigma$-conf. region', '$1\\sigma$-pred. band'], loc=4)
    fig.tight_layout()
    if plot_errbars and plot_samples:
        fig.savefig('pics/' + imagename + '_errbars_and_samples.pdf', dpi=225)
    elif plot_errbars:
        fig.savefig('pics/' + imagename + '_errbars.pdf', dpi=225)
    elif plot_samples:
        fig.savefig('pics/' + imagename + '_samples.pdf', dpi=225)

#==============================================================================
#               Далее идёт непосредственно скриптовая часть
#==============================================================================

#Загрузим каталог:
cat0 = pd.read_excel('./catalogues/work_catalog_2022.xlsx', usecols = 'A,C:H,M:O,X')
cat0 = cat0.replace('N', np.nan)
cat0 = cat0.replace(0, np.nan)
cat0 = cat0.replace('>100', 100)
cat0 = cat0.replace('>300', 300)

#Заменим неизвестные ошибки медианными:
def replace_nan_by_med(cat, colnames, sym=False):
    if sym:
        col, err = colnames
        #medval = np.median(cat[col][~np.isnan(cat[col])])
        mederr = np.median(cat[err][~np.isnan(cat[err])] / cat[col][~np.isnan(cat[err])])
        #cat.loc[np.isnan(cat[col]), col] = medval
        cat.loc[np.isnan(cat[err]), err] = mederr * cat[col][np.isnan(cat[err])]
    else:
        col, toplim, botlim = colnames
        toperr = ( cat[toplim][~np.isnan(cat[toplim])] - cat[col][~np.isnan(cat[toplim])] ) / cat[col][~np.isnan(cat[toplim])]
        boterr = ( cat[col][~np.isnan(cat[botlim])] - cat[botlim][~np.isnan(cat[botlim])] ) / cat[col][~np.isnan(cat[botlim])]
        #valmed = np.median(cat[col][~np.isnan(cat[col])])
        topmed = np.median(toperr)
        botmed = np.median(boterr)
        #cat.loc[np.isnan(cat[col]), col] = valmed
        cat.loc[np.isnan(cat[toplim]), toplim] = cat[col][np.isnan(cat[toplim])] * (1 + topmed)
        cat.loc[np.isnan(cat[botlim]), botlim] = cat[col][np.isnan(cat[botlim])] * (1 - botmed)

replace_nan_by_med(cat0, ('CPL:alpha', 'CPL:alpha+', 'CPL:alpha-'))
replace_nan_by_med(cat0, ('CPL:Ep', 'CPL:Ep+', 'CPL:Ep-'))
replace_nan_by_med(cat0, ('BAT Fluence\n(15-150 keV)\n[10-7 erg/cm2]', 'BAT Fluence\n90% Error\n(15-150 keV)\n[10-7 erg/cm2]'), sym=True)

#Найдём значения E_iso и E_p,i
#Сначала считаем наши наблюдаемые параметры и их погрешности
S_obs_arr = np.array(cat0['BAT Fluence\n(15-150 keV)\n[10-7 erg/cm2]'])
E_p_arr = np.array(cat0['CPL:Ep'])
E_p_arr_u = np.array(cat0['CPL:Ep+'])
nonans_mask = ~np.isnan(S_obs_arr) * (E_p_arr_u - E_p_arr > 0.0)
S_obs_arr = S_obs_arr[nonans_mask]
E_p_arr = E_p_arr[nonans_mask]
E_p_arr_u = E_p_arr_u[nonans_mask]
E_p_arr_d = np.array(cat0['CPL:Ep-'])[nonans_mask]
DS_obs_arr = np.array(cat0['BAT Fluence\n90% Error\n(15-150 keV)\n[10-7 erg/cm2]'])[nonans_mask] / norm.ppf(0.95)
z_arr = np.array(cat0['Redshift'])[nonans_mask]
alpha_arr = np.array(cat0['CPL:alpha'])[nonans_mask]
alpha_arr_u = np.array(cat0['CPL:alpha+'])[nonans_mask]
alpha_arr_d = np.array(cat0['CPL:alpha-'])[nonans_mask]
GRB_names = np.array(cat0['GRB'], dtype=str)[nonans_mask]
GRB_amount = np.sum(nonans_mask)

Dalpha_arr_u = alpha_arr_u - alpha_arr
Dalpha_arr_d = alpha_arr - alpha_arr_d
DE_p_arr_u = E_p_arr_u - E_p_arr
DE_p_arr_d = E_p_arr - E_p_arr_d

#Назначаем выборочные значения для альфы, Ep и Sobs, и сразу же находим значения x и y для плоскости Амати
print('Создание выборок для гамма-всплесков')
def loopfun_makesamples(i):
    alpha_sample, E_p_sample, S_obs_sample = make_samples(i, (alpha_arr, Dalpha_arr_d, Dalpha_arr_u, E_p_arr, DE_p_arr_d, DE_p_arr_u, S_obs_arr, DS_obs_arr, z_arr), amount = sample_size)
    Amx_sample, Amy_sample = sample_Amatixy(i, (alpha_sample, E_p_sample, S_obs_sample, z_arr))
    return (alpha_sample, E_p_sample, S_obs_sample, Amx_sample, Amy_sample)
    
alpha_all_samples, E_p_all_samples, S_obs_all_samples, Amx_all_samples, Amy_all_samples = np.array(list(zip(*Parallel(n_jobs=Nthreads, max_nbytes='2048M', verbose=10)(delayed(loopfun_makesamples)(i) for i in np.arange(GRB_amount)))))

#def plot_sample_hist(arg_all_samples, i, name, arg_arr, arg_dlim, arg_ulim):
#    fig, ax = plt.subplots()
#    xlim_d = arg_arr[i] - (arg_arr[i] - arg_dlim[i]) * 4
#    xlim_u = arg_arr[i] + (arg_ulim[i] - arg_arr[i]) * 4
#    bins = np.linspace(xlim_d, xlim_u, 51)
#    limmask = (arg_all_samples[i,:] > xlim_d) * (arg_all_samples[i,:] < xlim_u)
#    ax.hist(arg_all_samples[i,:][limmask], bins=bins, color='c', density=True, ec='dimgray', linewidth=0.25)
#    ax.axvline(arg_arr[i], c='k', zorder=9, linewidth=2)
#    ax.axvline(arg_dlim[i], c='k', zorder=9, linewidth=2, linestyle='dashed')
#    ax.axvline(arg_ulim[i], c='k', zorder=9, linewidth=2, linestyle='dashed')
#    ax.axvline(np.median(arg_all_samples[i,:]), c='darkblue', zorder=10, linewidth=2)
#    ax.axvline(np.percentile(arg_all_samples[i,:], 16), c='darkblue', zorder=10, linewidth=2, linestyle='dashed')
#    ax.axvline(np.percentile(arg_all_samples[i,:], 84), c='darkblue', zorder=10, linewidth=2, linestyle='dashed')
#    fig.savefig('./pics_temp/'+name+'_'+str(i)+'_split.png', dpi=144)
#    return 0

#plt.ioff()
#for i in np.arange(GRB_amount):
#    print(i)
#    plot_sample_hist(alpha_all_samples, i, 'alpha', alpha_arr, alpha_arr_d, alpha_arr_u)
#    plot_sample_hist(E_p_all_samples, i, 'Ep', E_p_arr, E_p_arr_d, E_p_arr_u)
#plt.ion()

#Для полученных выборок находим медианы, а также верхние и нижние пределы
Amx_meds, Amx_dlim, Amx_ulim = get_meds_and_lims(Amx_all_samples)
Amy_meds, Amy_dlim, Amy_ulim = get_meds_and_lims(Amy_all_samples)
#Amx_meds, Amx_dlim, Amx_ulim = get_twopiece_pars(Amx_all_samples)
#Amy_meds, Amy_dlim, Amy_ulim = get_twopiece_pars(Amy_all_samples)

#Спрямляем выборки
Amx_flat = Amx_all_samples.flatten()
Amy_flat = Amy_all_samples.flatten()

#Amx_lim_d = -0.75
#Amx_lim_u = 5.75
#Amy_lim_d = 40
#Amy_lim_u = 63.5
#limmask = (Amx_flat > Amx_lim_d) * (Amx_flat < Amx_lim_u) * (Amy_flat > Amy_lim_d) * (Amy_flat < Amy_lim_u)

#Для каждой реализации (их 1024 по умолчанию) находим коэффициенты a и b методом Тейла-Сена
a_sample = np.empty(sample_size)
b_sample = np.empty(sample_size)
for i in np.arange(sample_size):
    finitemask = np.isfinite(Amx_all_samples[:,i]) * np.isfinite(Amy_all_samples[:,i])
    fit_res = theilslopes(Amy_all_samples[:,i][finitemask], Amx_all_samples[:,i][finitemask])
    a_sample[i] = fit_res[0]
    b_sample[i] = fit_res[1]

#А здесь определяем среди этих выборок a и b медианы и погрешности
a_est = np.median(a_sample)
b_est = np.median(b_sample)
Da_est = (np.percentile(a_sample, 100 * err_qu) - np.percentile(a_sample, 100 * err_ql)) / 2
Db_est = (np.percentile(b_sample, 100 * err_qu) - np.percentile(b_sample, 100 * err_ql)) / 2

#Можем взглянуть на корреляцию между a и b:
fig = plt.figure(figsize=(5,5))
fig = corner.corner(np.array([a_sample, b_sample]).T, fig=fig, labels=[r'$a$', r'$b$'], quantiles=[err_ql, 0.5, err_qu], show_titles=True, bins=30)
fig.savefig('pics/cornerplot_ts.pdf', dpi=300)

#Далее нарисуем картинки:
finitemask = np.isfinite(Amx_flat) * np.isfinite(Amy_flat)
Amx_flat = Amx_flat[finitemask]
Amy_flat = Amy_flat[finitemask]

fig, ax = plt.subplots()
plot_Amati(fig, ax, (a_sample, b_sample, Amx_flat, Amy_flat, Amx_meds, Amy_meds), 'Amati_ts', 'repeated Theil-Sen estimation', plot_errbars=False, col2 = 'darkblue')
fig, ax = plt.subplots()
plot_Amati(fig, ax, (a_sample, b_sample, Amx_meds, Amy_meds, Amx_ulim, Amy_ulim, Amx_dlim, Amy_dlim), 'Amati_ts', 'repeated Theil-Sen estimation', plot_samples=False, col1='teal')
fig, ax = plt.subplots()
plot_Amati(fig, ax, (a_sample, b_sample, Amx_flat, Amy_flat, Amx_meds, Amy_meds, Amx_ulim, Amy_ulim, Amx_dlim, Amy_dlim), 'Amati_ts', 'repeated Theil-Sen estimation')

#Зная a и b, найдём mu_A:
print('Расчёт mu_A для выборок гамма-всплесков с оценкой a и b методом Theil-Sen')

def loopfun_sample_mu_A(i):
    return tuple(sample_mu_A(i, (alpha_all_samples[i,:], E_p_all_samples[i,:], S_obs_all_samples[i,:], z_arr), a_sample, b_sample, 0.0))

mu_A_all_samples = np.array(list(zip(*Parallel(n_jobs=Nthreads, max_nbytes='2048M', verbose=10)(delayed(loopfun_sample_mu_A)(i) for i in np.arange(GRB_amount))))).T

mu_A_meds, mu_A_dlim, mu_A_ulim = get_meds_and_lims(mu_A_all_samples)
z_log_range = 10 ** np.linspace(np.min(np.log10(z_arr)), np.max(np.log10(z_arr)), 101)
mu_arr_st = np.empty(101)
for i in np.arange(101):
    mu_arr_st[i] = mu(z_log_range[i], pars_cosm_planck70)

#Нарисуем теперь диаграмму Хаббла:
fig, ax = plt.subplots()
ax.plot(z_log_range, mu_arr_st, c='k', zorder=10)
ax.errorbar(z_arr, mu_A_meds, yerr=np.array([mu_A_dlim, mu_A_ulim]), linestyle='', linewidth=0.3, marker='o', markersize=1.5, c='teal')
ax.set_xscale('log')
ax.set_ylim((32, 52))
ax.set_title('GRB Hubble Diagram')
ax.set_xlabel(r'$z$')
ax.set_ylabel(r'$\mu$')
fig.tight_layout()
fig.savefig('pics/HD_ts.png', dpi=300)

#Определим функцию, вычисляющую отклонения для inverse cosmological fitting
def lcdm_inv_residuals(pars, z_arg, alpha_arg, E_p_arg, S_obs_arg, Amx_arg, Amy_arg):
    mu_A_inner = mu_A_ufunc(z_arg, alpha_arg, E_p_arg, S_obs_arg, pars[0], pars[1], pars[2])
    infinitemask_inner = ~np.isfinite(mu_A_inner)
    mu_A_inner[infinitemask_inner] = 0.0
    Amy_arg[~np.isfinite(Amy_arg)] = 0.0
    Amx_arg[~np.isfinite(Amx_arg)] = 0.0
    residuals_HD = (mu_A_inner - mu_SN_ufunc(z_arg))
    residuals_Am = (Amy_arg - pars[0] * Amx_arg - pars[1]) * 2.2  #Домножаем на 2.2, чтобы отклонения были того же масштаба
    return np.concatenate((residuals_HD, residuals_Am))# np.repeat(residuals_Am, 5)))

#Проводим inverse cosmological fitting робастным методом
print('Вычисление коэффициентов a, b и k методом icf')
def loopfun_icf(i):
    return tuple(least_squares(lcdm_inv_residuals, (1.0, 50.0, 0.0), loss='soft_l1', f_scale=2.5, args=(z_arr, alpha_all_samples[:,i], E_p_all_samples[:,i], S_obs_all_samples[:,i], Amx_all_samples[:,i], Amy_all_samples[:,i])).x)

a_inv_sample, b_inv_sample, k_inv_sample = np.array(list(zip(*Parallel(n_jobs=Nthreads, max_nbytes='2048M', verbose=10)(delayed(loopfun_icf)(i) for i in np.arange(sample_size)))))

fig = plt.figure(figsize=(6,6))
corner.corner(np.array([a_inv_sample, b_inv_sample, k_inv_sample]).T, fig=fig, labels=[r'$a$', r'$b$', r'$k$'], quantiles=[err_ql, 0.5, err_qu], show_titles=True, bins=30)
fig.savefig('pics/cornerplot_icf.png', dpi=300)

a_inv_est = np.median(a_inv_sample)
Da_inv = (np.percentile(a_inv_sample, 100 * err_qu) - np.percentile(a_inv_sample, 100 * err_ql)) / 2
b_inv_est = np.median(b_inv_sample)
Db_inv = (np.percentile(b_inv_sample, 100 * err_qu) - np.percentile(b_inv_sample, 100 * err_ql)) / 2
k_inv_est = np.median(k_inv_sample)
Dk_inv = (np.percentile(k_inv_sample, 100 * err_qu) - np.percentile(k_inv_sample, 100 * err_ql)) / 2

#Определим функцию, вычисляющую отклонения для inverse cosmological fitting с k=0:
def lcdm_inv_residuals_k0(pars, z_arg, alpha_arg, E_p_arg, S_obs_arg, Amx_arg, Amy_arg):
    mu_A_inner = mu_A_ufunc(z_arg, alpha_arg, E_p_arg, S_obs_arg, pars[0], pars[1], 0.0)
    infinitemask_inner = ~np.isfinite(mu_A_inner)
    mu_A_inner[infinitemask_inner] = 0.0
    Amy_arg[~np.isfinite(Amy_arg)] = 0.0
    Amx_arg[~np.isfinite(Amx_arg)] = 0.0
    residuals_HD = (mu_A_inner - mu_SN_ufunc(z_arg))
    residuals_Am = (Amy_arg - pars[0] * Amx_arg - pars[1]) * 2.2  #Домножаем на 2.2, чтобы отклонения были того же масштаба
    return np.concatenate((residuals_HD, residuals_Am))# np.repeat(residuals_Am, 5)))

print('Вычисление коэффициентов a, b методом icf, k=0')
def loopfun_icf_k0(i):
    return tuple(least_squares(lcdm_inv_residuals_k0, (1.0, 50.0), loss='soft_l1', f_scale=2.5, args=(z_arr, alpha_all_samples[:,i], E_p_all_samples[:,i], S_obs_all_samples[:,i], Amx_all_samples[:,i], Amy_all_samples[:,i])).x)

a_inv_k0_sample, b_inv_k0_sample = np.array(list(zip(*Parallel(n_jobs=Nthreads, max_nbytes='2048M', verbose=10)(delayed(loopfun_icf_k0)(i) for i in np.arange(sample_size)))))

fig = plt.figure(figsize=(5,5))
corner.corner(np.array([a_inv_k0_sample, b_inv_k0_sample]).T, fig=fig, labels=[r'$a$', r'$b$'], quantiles=[err_ql, 0.5, err_qu], show_titles=True, bins=30)
fig.savefig('pics/cornerplot_icf_k0.png', dpi=300)

a_inv_k0_est = np.median(a_inv_k0_sample)
Da_inv_k0 = (np.percentile(a_inv_k0_sample, 100 * err_qu) - np.percentile(a_inv_k0_sample, 100 * err_ql)) / 2
b_inv_k0_est = np.median(b_inv_k0_sample)
Db_inv_k0 = (np.percentile(b_inv_k0_sample, 100 * err_qu) - np.percentile(b_inv_k0_sample, 100 * err_ql)) / 2

def lcdm_residuals_chi2(pars, z_arg, alpha_arg, E_p_arg, S_obs_arg, Amx_arg, Amy_arg):
    mu_A_inner = mu_A_ufunc(z_arg, alpha_arg, E_p_arg, S_obs_arg, pars[0], pars[1], pars[2])
    infinitemask_inner = ~np.isfinite(mu_A_inner)
    mu_A_inner[infinitemask_inner] = 0.0
    Amy_arg[~np.isfinite(Amy_arg)] = 0.0
    Amx_arg[~np.isfinite(Amx_arg)] = 0.0
    residuals_HD = (mu_A_inner - mu_SN_ufunc(z_arg)) ** 2 / mu_SN_ufunc(z_arg)
    return np.sqrt(np.sum(residuals_HD))

#Теперь определим mu_A с помощью новых a и b
print('Расчёт mu_A для выборок гамма-всплесков с коэффициентами a и b, полученными обратной калибровкой')
def loopfun_sample_mu_A_inv(i):
    return tuple(sample_mu_A(i, (alpha_all_samples[i,:], E_p_all_samples[i,:], S_obs_all_samples[i,:], z_arr), a_inv_sample, b_inv_sample, k_inv_sample))

def loopfun_sample_mu_A_inv_k0(i):
    return tuple(sample_mu_A(i, (alpha_all_samples[i,:], E_p_all_samples[i,:], S_obs_all_samples[i,:], z_arr), a_inv_k0_sample, b_inv_k0_sample, 0.0))


mu_A_inv_all_samples = np.array(list(zip(*Parallel(n_jobs=Nthreads, max_nbytes='2048M', verbose=10)(delayed(loopfun_sample_mu_A_inv)(i) for i in np.arange(GRB_amount))))).T
mu_A_inv_k0_all_samples = np.array(list(zip(*Parallel(n_jobs=Nthreads, max_nbytes='2048M', verbose=10)(delayed(loopfun_sample_mu_A_inv_k0)(i) for i in np.arange(GRB_amount))))).T
mu_A_inv_all_samples[~np.isfinite(mu_A_inv_all_samples)] = 0.0
mu_A_inv_k0_all_samples[~np.isfinite(mu_A_inv_k0_all_samples)] = 0.0

mu_A_inv_meds, mu_A_inv_dlim, mu_A_inv_ulim = get_meds_and_lims(mu_A_inv_all_samples)
mu_A_inv_k0_meds, mu_A_inv_k0_dlim, mu_A_inv_k0_ulim = get_meds_and_lims(mu_A_inv_k0_all_samples)

#Снова нарисуем диаграмму Хаббла
fig, ax = plt.subplots()
ax.plot(z_log_range, mu_arr_st, c='k', zorder=10)
ax.errorbar(z_arr, mu_A_inv_meds, yerr=np.array([mu_A_inv_dlim, mu_A_inv_ulim]), linestyle='', linewidth=0.3, marker='o', markersize=1.25, c='teal')
ax.set_xscale('log')
ax.set_ylim((32, 52))
ax.set_title('GRB Hubble Diagram (free $k$)')
ax.set_xlabel(r'$z$')
ax.set_ylabel(r'$\mu$')
fig.tight_layout()
fig.savefig('pics/HD_icf.png', dpi=300)

fig, ax = plt.subplots()
ax.plot(z_log_range, mu_arr_st, c='k', zorder=10)
ax.errorbar(z_arr, mu_A_inv_k0_meds, yerr=np.array([mu_A_inv_dlim, mu_A_inv_ulim]), linestyle='', linewidth=0.3, marker='o', markersize=1.25, c='teal')
ax.set_xscale('log')
ax.set_ylim((32, 52))
ax.set_title('GRB Hubble Diagram ($k=0$)')
ax.set_xlabel(r'$z$')
ax.set_ylabel(r'$\mu$')
fig.tight_layout()
fig.savefig('pics/HD_icf_k0.png', dpi=300)

#И плоскость Амати
fig, ax = plt.subplots()
plot_Amati(fig, ax, (a_inv_sample, b_inv_sample, Amx_flat, Amy_flat, Amx_meds, Amy_meds), 'Amati_icf', 'inv. cosm. fitting (free $k$)', plot_errbars=False, col2 = 'darkblue')
fig, ax = plt.subplots()
plot_Amati(fig, ax, (a_inv_sample, b_inv_sample, Amx_meds, Amy_meds, Amx_ulim, Amy_ulim, Amx_dlim, Amy_dlim), 'Amati_icf', 'inv. cosm. fitting (free $k$)', plot_samples=False, col1='teal')
fig, ax = plt.subplots()
plot_Amati(fig, ax, (a_inv_sample, b_inv_sample, Amx_flat, Amy_flat, Amx_meds, Amy_meds, Amx_ulim, Amy_ulim, Amx_dlim, Amy_dlim), 'Amati_icf', 'inv. cosm. fitting (free $k$)')

fig, ax = plt.subplots()
plot_Amati(fig, ax, (a_inv_k0_sample, b_inv_k0_sample, Amx_flat, Amy_flat, Amx_meds, Amy_meds), 'Amati_icf_k0', 'inv. cosm. fitting ($k=0$)', plot_errbars=False, col2 = 'darkblue')
fig, ax = plt.subplots()
plot_Amati(fig, ax, (a_inv_k0_sample, b_inv_k0_sample, Amx_meds, Amy_meds, Amx_ulim, Amy_ulim, Amx_dlim, Amy_dlim), 'Amati_icf_k0', 'inv. cosm. fitting ($k=0$)', plot_samples=False, col1='teal')
fig, ax = plt.subplots()
plot_Amati(fig, ax, (a_inv_k0_sample, b_inv_k0_sample, Amx_flat, Amy_flat, Amx_meds, Amy_meds, Amx_ulim, Amy_ulim, Amx_dlim, Amy_dlim), 'Amati_icf_k0', 'inv. cosm. fitting ($k=0$)')

print('Вычисление хи-квадрат для медианных параметров:')
print('Для случая варьирования a, b, k: ', str(lcdm_residuals_chi2((a_inv_est, b_inv_est, k_inv_est), z_arr, alpha_arr, E_p_arr, S_obs_arr, Amx_meds, Amy_meds)))
print('Для случая варьирования a, b с фиксированным k=0: ', str(lcdm_residuals_chi2((a_inv_k0_est, b_inv_k0_est, 0.0), z_arr, alpha_arr, E_p_arr, S_obs_arr, Amx_meds, Amy_meds)))

#==============================================================================
#                         Загрузим каталог Pantheon
#==============================================================================

catSN = pd.read_csv('catalogues/Pantheon.dat', delimiter='\t', header=0, usecols = [2, 4, 5])

z_arr_SN = np.array(catSN['zcmb'])
mu_arr_SN = np.array(catSN['mu'])# + 19.41623729
Dmu_arr_SN = np.array(catSN['err_mu'])

z_log_range = 10 ** np.linspace(np.log10(np.min(z_arr_SN)), np.log10(np.max(z_arr_SN)))

#Определим модели:

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
    logz=np.log10(1+z)
    return 5 * np.log10(c / H0 * z) + 25 + a1 * logz + a2 * logz ** 2

@jit(nopython=True)
def theor_just3(z, a1, a2, a3):
    logz=np.log10(1+z)
    return 5 * np.log10(c / H0 * z) + 25 + a1 * logz + a2 * logz ** 2 + a3 * logz ** 3

@jit(nopython=True)
def theor_just4(z, a1, a2, a3, a4):
    logz=np.log10(1+z)
    return 5 * np.log10(c / H0 * z) + 25 + a1 * logz + a2 * logz ** 2 + a3 * logz ** 3 + a4 * logz ** 4

@jit(nopython=True)
def theor_just5(z, a1, a2, a3, a4, a5):
    logz=np.log10(1+z)
    return 5 * np.log10(c / H0 * z) + 25 + a1 * logz + a2 * logz ** 2 + a3 * logz ** 3 + a4 * logz ** 4 + a5 * logz ** 5

@jit(nopython=True)
def offpolylog2(z, a0, a1, a2):
    logz = np.log10(1+z)
    return a0 + a1 * logz + a2 * logz ** 2

@jit(nopython=True)
def offpolylog3(z, a0, a1, a2, a3):
    logz = np.log10(1+z)
    return a0 + a1 * logz + a2 * logz ** 2 + a3 * logz ** 3

@jit(nopython=True)
def offpolylog4(z, a0, a1, a2, a3, a4):
    logz = np.log10(1+z)
    return a0 + a1 * logz + a2 * logz ** 2 + a3 * logz ** 3 + a4 * logz ** 4

@jit(nopython=True)
def offpolylog5(z, a0, a1, a2, a3, a4, a5):
    logz = np.log10(1+z)
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
polylog5_best = lambda z: polylog5(z, fit_pl5[0][0], fit_pl5[0][1], fit_pl5[0][2], fit_pl5[0][3], fit_pl5[0][4], fit_pl5[0][5])
theor_just2_best = lambda z: theor_just2(z, fit_tj2[0][0], fit_tj2[0][1])
theor_just3_best = lambda z: theor_just3(z, fit_tj3[0][0], fit_tj3[0][1], fit_tj3[0][2])
theor_just4_best = lambda z: theor_just4(z, fit_tj4[0][0], fit_tj4[0][1], fit_tj4[0][2], fit_tj4[0][3])
theor_just5_best = lambda z: theor_just5(z, fit_tj5[0][0], fit_tj5[0][1], fit_tj5[0][2], fit_tj5[0][3], fit_tj5[0][4])
offpolylog2_best = lambda z: offpolylog2(z, fit_opl2[0][0], fit_opl2[0][1], fit_opl2[0][2])
offpolylog3_best = lambda z: offpolylog3(z, fit_opl3[0][0], fit_opl3[0][1], fit_opl3[0][2], fit_opl3[0][3])
offpolylog4_best = lambda z: offpolylog4(z, fit_opl4[0][0], fit_opl4[0][1], fit_opl4[0][2], fit_opl4[0][3], fit_opl4[0][4])
offpolylog5_best = lambda z: offpolylog5(z, fit_opl5[0][0], fit_opl5[0][1], fit_opl5[0][2], fit_opl5[0][3], fit_opl5[0][4], fit_opl5[0][5])

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
    return '$\mathrm{AIC}='+str(np.around(AIC, 1))+',\;\mathrm{BIC}='+str(np.around(BIC, 1))+'$'

fig, ax = plt.subplots(figsize=(7,5.5))
ax.plot(z_log_range, mu_range_pl1, zorder=5, c='gold')
ax.plot(z_log_range, mu_range_pl2, zorder=5)
ax.plot(z_log_range, mu_range_pl3, zorder=5)
ax.plot(z_log_range, mu_range_pl4, zorder=5)
ax.plot(z_log_range, mu_range_pl5, zorder=5)
#ax.axvline(z_border, c='k', linestyle='dashed')
ax.set_xlabel('redshift $z$')
ax.set_ylabel('distance modulus $\\mu$')
ax.set_title('Pantheon SN Ia Hubble diagram')
ax.legend(['Polylog., $p=1$, '+legend_IC(AIC_polylog1, BIC_polylog1), 'Polylog., $p=2$, '+legend_IC(AIC_polylog2, BIC_polylog2), 'Polylog., $p=3$, '+legend_IC(AIC_polylog3, BIC_polylog3), 'Polylog., $p=4$, '+legend_IC(AIC_polylog4, BIC_polylog4), 'Polylog., $p=5$, '+legend_IC(AIC_polylog5, BIC_polylog5)], fancybox=True, shadow=True)
ax.errorbar(z_arr_SN, mu_arr_SN, Dmu_arr_SN, color='silver', linestyle='', marker='o', markeredgewidth=0.25, markeredgecolor='k', markersize=3.5, rasterized=True)
ax.set_xscale('log')
fig.tight_layout()
fig.savefig('./pics_misc/SNHD_polylog.pdf', dpi=225)

fig, ax = plt.subplots(figsize=(7,5.5))
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
ax.legend(['Linear Hubble law', 'Theoretically justified, $p=2$, '+legend_IC(AIC_theor_just2, BIC_theor_just2), 'Theoretically justified, $p=3$, '+legend_IC(AIC_theor_just3, BIC_theor_just3), 'Theoretically justified, $p=4$, '+legend_IC(AIC_theor_just4, BIC_theor_just4), 'Theoretically justified, $p=5$, '+legend_IC(AIC_theor_just5, BIC_theor_just5)], fancybox=True, shadow=True)
ax.errorbar(z_arr_SN, mu_arr_SN, Dmu_arr_SN, color='silver', linestyle='', marker='o', markeredgewidth=0.25, markeredgecolor='k', markersize=3.5, rasterized=True)
ax.set_xscale('log')
fig.tight_layout()
fig.savefig('./pics_misc/SNHD_theor_just.pdf', dpi=300)

fig, ax = plt.subplots(figsize=(7,5.5))
ax.plot(z_log_range, mu_range_opl2, zorder=5)
ax.plot(z_log_range, mu_range_opl3, zorder=5)
ax.plot(z_log_range, mu_range_opl4, zorder=5)
ax.plot(z_log_range, mu_range_opl5, zorder=5)
ax.set_xlabel('redshift $z$')
ax.set_ylabel('distance modulus $\\mu$')
ax.set_title('Pantheon SN Ia Hubble diagram')
ax.legend(['Offset polylogarithmic, $p=2$, '+legend_IC(AIC_offpolylog2, BIC_offpolylog2), 'Offset polylogarithmic, $p=3$, '+legend_IC(AIC_offpolylog3, BIC_offpolylog3), 'Offset polylogarithmic, $p=4$, '+legend_IC(AIC_offpolylog4, BIC_offpolylog4), 'Offset polylogarithmic, $p=5$, '+legend_IC(AIC_offpolylog5, BIC_offpolylog5)], fancybox=True, shadow=True)
ax.errorbar(z_arr_SN, mu_arr_SN, Dmu_arr_SN, color='silver', linestyle='', marker='o', markeredgewidth=0.25, markeredgecolor='k', markersize=3.5, rasterized=True)
ax.set_xscale('log')
fig.tight_layout()
fig.savefig('./pics_misc/SNHD_offpolylog.pdf', dpi=300)

#Самая лучшая модель -- полилогарифмическая с p=3. Создадим также Монте_Карло выборки для её параметров:

a1_sample, a2_sample = multivariate_normal.rvs(fit_tj2[0], fit_tj2[1], sample_size).T
fig = plt.figure(figsize=(5, 5))
fig = corner.corner(np.array([a1_sample, a2_sample]).T, fig=fig, labels=[r'$a_1$', r'$a_2$'], quantiles=[err_ql, 0.5, err_qu], show_titles=True, bins=25)
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
    Amy_sample = np.vectorize(lambda z, alpha, E_p, S_obs, a1, a2: Amy_SN_tj2(z, alpha, E_p, S_obs, a1, a2))(z_arg[ii], alpha_sample, E_p_sample, S_obs_sample, a1_s, a2_s)
    Amy_sample[np.isnan(Amy_sample)] = np.inf
    return (Amx_sample, Amy_sample)

#Находим значения x и y для плоскости Амати
print('Создание выборок для гамма-всплесков')
def loopfun_makesamples_SN_tj2(i):
    Amx_sample_SN_tj2, Amy_sample_SN_tj2 = sample_Amatixy_SN_tj2(i, (alpha_all_samples[i,:], E_p_all_samples[i,:], S_obs_all_samples[i,:], z_arr, a1_sample, a2_sample))
    return (Amx_sample_SN_tj2, Amy_sample_SN_tj2)

Amx_all_samples_SN_tj2, Amy_all_samples_SN_tj2 = np.array(list(zip(*Parallel(n_jobs=Nthreads, max_nbytes='2048M', verbose=10)(delayed(loopfun_makesamples_SN_tj2)(i) for i in np.arange(GRB_amount)[z_arr < 1.4]))))

#Для полученных выборок находим медианы, а также верхние и нижние пределы
Amx_meds_SN_tj2, Amx_dlim_SN_tj2, Amx_ulim_SN_tj2 = get_meds_and_lims(Amx_all_samples_SN_tj2)
Amy_meds_SN_tj2, Amy_dlim_SN_tj2, Amy_ulim_SN_tj2 = get_meds_and_lims(Amy_all_samples_SN_tj2)

#Спрямляем выборки
Amx_flat_SN_tj2 = Amx_all_samples_SN_tj2.flatten()
Amy_flat_SN_tj2 = Amy_all_samples_SN_tj2.flatten()

#Amx_lim_d = -0.75
#Amx_lim_u = 5.75
#Amy_lim_d = 40
#Amy_lim_u = 63.5
#limmask = (Amx_flat > Amx_lim_d) * (Amx_flat < Amx_lim_u) * (Amy_flat > Amy_lim_d) * (Amy_flat < Amy_lim_u)

#Для каждой реализации (их 1024 по умолчанию) находим коэффициенты a и b методом Тейла-Сена
a_sample_SN_tj2 = np.empty(sample_size)
b_sample_SN_tj2 = np.empty(sample_size)
for i in np.arange(sample_size):
    finitemask = np.isfinite(Amx_all_samples_SN_tj2[:,i]) * np.isfinite(Amy_all_samples_SN_tj2[:,i])
    fit_res = theilslopes(Amy_all_samples_SN_tj2[:,i][finitemask], Amx_all_samples_SN_tj2[:,i][finitemask])
    a_sample_SN_tj2[i] = fit_res[0]
    b_sample_SN_tj2[i] = fit_res[1]

#А здесь определяем среди этих выборок a и b медианы и погрешности
a_est_SN_tj2 = np.median(a_sample_SN_tj2)
b_est_SN_tj2 = np.median(b_sample_SN_tj2)
Da_est_SN_tj2 = (np.percentile(a_sample_SN_tj2, 100 * err_qu) - np.percentile(a_sample_SN_tj2, 100 * err_ql)) / 2
Db_est_SN_tj2 = (np.percentile(b_sample_SN_tj2, 100 * err_qu) - np.percentile(b_sample_SN_tj2, 100 * err_ql)) / 2

#Можем взглянуть на корреляцию между a и b:
fig = plt.figure(figsize=(5,5))
fig = corner.corner(np.array([a_sample_SN_tj2, b_sample_SN_tj2]).T, fig=fig, labels=[r'$a$', r'$b$'], quantiles=[err_ql, 0.5, err_qu], show_titles=True, bins=30)
fig.savefig('pics/cornerplot_ts_SN_tj2.pdf', dpi=225)

#fig = plt.figure(figsize=(5,5))
#fig = corner.corner(np.array([a_sample_SN_pl3, b_sample_SN_pl3]).T, fig=fig, labels=[r'$a$', r'$b$'], quantiles=[], show_titles=True, bins=30, color='Blue')
#corner.corner(np.array([a_sample, b_sample]).T, fig=fig, bins=30)
#fig.savefig('pics/cornerplot_ts_multiple.png', dpi=300)


#Далее нарисуем картинки:
finitemask = np.isfinite(Amx_flat_SN_tj2) * np.isfinite(Amy_flat_SN_tj2)
Amx_flat_SN_tj2 = Amx_flat_SN_tj2[finitemask]
Amy_flat_SN_tj2 = Amy_flat_SN_tj2[finitemask]

fig, ax = plt.subplots()
plot_Amati(fig, ax, (a_sample_SN_tj2, b_sample_SN_tj2, Amx_flat_SN_tj2, Amy_flat_SN_tj2, Amx_meds_SN_tj2, Amy_meds_SN_tj2), 'Amati_ts_SN_tj2', 'repeated Theil-Sen estimation', plot_errbars=False, col2 = 'darkblue')
fig, ax = plt.subplots()
plot_Amati(fig, ax, (a_sample_SN_tj2, b_sample_SN_tj2, Amx_meds_SN_tj2, Amy_meds_SN_tj2, Amx_ulim_SN_tj2, Amy_ulim_SN_tj2, Amx_dlim_SN_tj2, Amy_dlim_SN_tj2), 'Amati_ts_SN_tj2', 'repeated Theil-Sen estimation', plot_samples=False, col1='teal')
fig, ax = plt.subplots()
plot_Amati(fig, ax, (a_sample_SN_tj2, b_sample_SN_tj2, Amx_flat_SN_tj2, Amy_flat_SN_tj2, Amx_meds_SN_tj2, Amy_meds_SN_tj2, Amx_ulim_SN_tj2, Amy_ulim_SN_tj2, Amx_dlim_SN_tj2, Amy_dlim_SN_tj2), 'Amati_ts_SN_tj2', 'repeated Theil-Sen estimation')

#Зная a и b, найдём mu_A:
print('Расчёт mu_A для выборок гамма-всплесков с оценкой a и b методом Theil-Sen')

def loopfun_sample_mu_A_SN_tj2(i):
    return tuple(sample_mu_A(i, (alpha_all_samples[i,:], E_p_all_samples[i,:], S_obs_all_samples[i,:], z_arr), a_sample_SN_tj2, b_sample_SN_tj2, 0.0))

mu_A_all_samples_SN_tj2 = np.array(list(zip(*Parallel(n_jobs=Nthreads, max_nbytes='2048M', verbose=10)(delayed(loopfun_sample_mu_A_SN_tj2)(i) for i in np.arange(GRB_amount))))).T

mu_A_meds_SN_tj2, mu_A_dlim_SN_tj2, mu_A_ulim_SN_tj2 = get_meds_and_lims(mu_A_all_samples_SN_tj2)
z_log_range = 10 ** np.linspace(np.min(np.log10(z_arr)), np.max(np.log10(z_arr)), 101)
mu_arr_st_SN_pl3 = np.empty(101)
for i in np.arange(101):
    mu_arr_st_SN_pl3[i] = mu(z_log_range[i], pars_cosm_planck70)

#Нарисуем теперь диаграмму Хаббла:
fig, ax = plt.subplots()
ax.plot(z_log_range, mu_arr_st, c='k', zorder=10)
ax.errorbar(z_arr, mu_A_meds_SN_tj2, yerr=np.array([mu_A_dlim_SN_tj2, mu_A_ulim_SN_tj2]), linestyle='', linewidth=0.3, marker='o', markersize=1.5, c='teal')
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
ax.errorbar(z_arr[z_arr < z_border], mu_A_meds_SN_tj2[z_arr < z_border], yerr=np.array([mu_A_dlim_SN_tj2[z_arr < z_border], mu_A_ulim_SN_tj2[z_arr < z_border]]), linestyle='', markeredgewidth=0.25, markeredgecolor='k', linewidth=0.3, marker='o', markersize=3, c='goldenrod', zorder=4, rasterized=True)
ax.errorbar(z_arr[z_arr >= z_border], mu_A_meds_SN_tj2[z_arr >= z_border], yerr=np.array([mu_A_dlim_SN_tj2[z_arr >= z_border], mu_A_ulim_SN_tj2[z_arr >= z_border]]), linestyle='', markeredgewidth=0.25, markeredgecolor='k', linewidth=0.3, marker='o', markersize=3, c='teal', zorder=3, rasterized=True)
ax.errorbar(z_arr_SN, mu_arr_SN, Dmu_arr_SN, color='silver', linestyle='', marker='o', markeredgewidth=0.25, markeredgecolor='k', markersize=2.5, rasterized=True)
ax.plot(z_log_range_big, mu_range_lcdm70, c='k', zorder=10)
ax.plot(z_log_range_big, 5 * np.log10(c / H0 * z_log_range_big) + 25, c='k', linewidth='1', linestyle='dashed', zorder=9)
ax_low.errorbar(z_arr[z_arr < z_border], Delta_mu_A_meds_SN_tj2[z_arr < z_border], yerr=np.array([mu_A_dlim_SN_tj2[z_arr < z_border], mu_A_ulim_SN_tj2[z_arr < z_border]]), linestyle='', markeredgewidth=0.25, markeredgecolor='k', linewidth=0.3, marker='o', markersize=3, c='goldenrod', zorder=4, rasterized=True)
ax_low.errorbar(z_arr[z_arr >= z_border], Delta_mu_A_meds_SN_tj2[z_arr >= z_border], yerr=np.array([mu_A_dlim_SN_tj2[z_arr >= z_border], mu_A_ulim_SN_tj2[z_arr >= z_border]]), linestyle='', markeredgewidth=0.25, markeredgecolor='k', linewidth=0.3, marker='o', markersize=3, c='teal', zorder=3, rasterized=True)
ax_low.errorbar(z_arr_SN, Delta_mu_arr_SN, Dmu_arr_SN, color='silver', linestyle='', marker='o', markeredgewidth=0.25, markeredgecolor='k', markersize=2.5, rasterized=True)
ax_low.plot(z_log_range_big, np.zeros(z_log_range_big.size), c='k', zorder=10)
ax_low.plot(z_log_range_big, - mu_range_lcdm70 + 5 * np.log10(c / H0 * z_log_range_big) + 25, c='k', linewidth='1', linestyle='dashed', zorder=9)
ax.legend(['$\Lambda$CDM', 'Linear Hubble Law', 'LGRBs with $z < 1.4$', 'LGRBs with $z \geq 1.4$', 'Pantheon SNe Ia'], fancybox=True, shadow=True)
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

fig, ([axlt, axrt], [axlb, axrb]) = plt.subplots(2, 2, sharex='col', sharey='row', gridspec_kw={'height_ratios': [4, 1]}, figsize=(12.5, 7))
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
axlt.legend(['Linear Hubble law', 'Theor. just., $p=2$, '+legend_IC(AIC_theor_just2, BIC_theor_just2), 'Theor. just., $p=3$, '+legend_IC(AIC_theor_just3, BIC_theor_just3), 'Theor. just., $p=4$, '+legend_IC(AIC_theor_just4, BIC_theor_just4), 'Theor. just., $p=5$, '+legend_IC(AIC_theor_just5, BIC_theor_just5)], fancybox=True, shadow=True)
axlt.errorbar(z_arr_SN, mu_arr_SN, Dmu_arr_SN, color='silver', linestyle='', marker='o', markeredgewidth=0.25, markeredgecolor='k', markersize=3.5, rasterized=True)
axlt.set_xscale('log')

mu_tj2_SN_range = theor_just2_best(z_arr_SN)

axlb.set_ylim((-0.75, 0.75))
axlb.plot(z_log_range, 5 * np.log10(c / H0 * z_log_range) + 25, zorder=5, c='black')
axlb.plot(z_log_range, mu_range_tj2-mu_range_tj2, zorder=6, c='tab:red')
axlb.plot(z_log_range, mu_range_tj3-mu_range_tj2, zorder=5)
axlb.plot(z_log_range, mu_range_tj4-mu_range_tj2, zorder=5)
axlb.plot(z_log_range, mu_range_tj5-mu_range_tj2, zorder=5)
axlb.axvline(z_border, c='k', linestyle='dashed')
axlb.errorbar(z_arr_SN, mu_arr_SN-mu_tj2_SN_range, Dmu_arr_SN, color='silver', linestyle='', marker='o', markeredgewidth=0.25, markeredgecolor='k', markersize=3.5, rasterized=True)
axlb.set_ylabel('$\\Delta\\mu$')

mu_lcdm_range = np.vectorize(lambda z: mu(z, pars_cosm_planck70))(z_log_range)

axrt.plot(z_log_range, mu_lcdm_range, zorder=5, c='black')
axrt.plot(z_log_range, mu_range_tj2, zorder=5, c='tab:red')
axrt.axvline(z_border, c='k', linestyle='dashed')
axrb.set_xlabel('redshift $z$')
axrt.set_title('Pantheon SN Ia Hubble diagram (linear scale)')
axrt.legend(['$\Lambda$CDM model', 'Theor. just., $p=2$'], loc='upper left', fancybox=True, shadow=True)
axrt.errorbar(z_arr_SN, mu_arr_SN, Dmu_arr_SN, color='silver', linestyle='', marker='o', markeredgewidth=0.25, markeredgecolor='k', markersize=3.5, rasterized=True)
axrt.set_xscale('linear')

axrb.plot(z_log_range, mu_lcdm_range-mu_range_tj2, zorder=5, c='black')
axrb.plot(z_log_range, mu_range_tj2-mu_range_tj2, zorder=5, c='tab:red')
axrb.axvline(z_border, c='k', linestyle='dashed')
axrb.errorbar(z_arr_SN, mu_arr_SN-mu_tj2_SN_range, Dmu_arr_SN, color='silver', linestyle='', marker='o', markeredgewidth=0.25, markeredgecolor='k', markersize=3.5, rasterized=True)

fig.tight_layout()
fig.savefig('./pics_misc/SNHD_theor_just_big.pdf')

# Для картинки со спектром:
@jit(nopython=True)
def E2NE(E, E_p, alpha):
    return E ** (2 + alpha) * np.exp(-(2 + alpha) * E / E_p) / (np.exp(-(2 + alpha) / E_p)) 

def E2NE_band(E_range, E_p_sample, alpha_sample, conf_int=100 * (err_qu - err_ql)):
    upper_bord = np.vectorize(lambda E: np.percentile(E ** (2 + alpha_sample) * np.exp(-(2 + alpha_sample) * E / E_p_sample) / (np.exp(-(2 + np.median(alpha_sample)) / np.median(E_p_sample))), (100 + conf_int) / 2))(E_range)
    lower_bord = np.vectorize(lambda E: np.percentile(E ** (2 + alpha_sample) * np.exp(-(2 + alpha_sample) * E / E_p_sample) / (np.exp(-(2 + np.median(alpha_sample)) / np.median(E_p_sample))), (100 - conf_int) / 2))(E_range)
    return (lower_bord, upper_bord)

E_range_specplot = 10 ** np.linspace(0, 3, 100)
E2NE_range_specplot = E2NE(E_range_specplot, E_p_arr[100], alpha_arr[100])
E2NE_range_1sig_d, E2NE_range_1sig_u = E2NE_band(E_range_specplot, E_p_all_samples[100,:], alpha_all_samples[100,:])

fig, ax = plt.subplots(figsize=(5.5, 4))
ax.grid(linestyle='dashed')
ax.plot(E_range_specplot, E2NE_range_specplot, zorder=20)
ax.fill_between(E_range_specplot, E2NE_range_1sig_d, E2NE_range_1sig_u, alpha=0.2, color='k', linestyle=':', linewidth=0.75)
ax.axvline(E_p_arr[100], c='red')
ax.axvline(E_p_arr[100] / (2 + alpha_arr[100]), c='k', linestyle='dashed')
ax.set_ylim((10 ** -1.6, 100))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Energy $E,\\,\\mathrm{keV}$')
ax.set_ylabel('$E^2N(E)$')
ax.legend(['spectrum', '$E_\\mathrm{p}$', '$E_\\mathrm{p} / (2 + \\alpha)$', '$1\\sigma$-conf. int. for spectrum'], labelspacing = 0.15, fancybox=True, shadow=True)
ax.set_title('The GRB'+GRB_names[100]+' spectrum at $z = {:.5}$'.format(z_arr[100])+',\n'+'$E_\\mathrm{p}'+' = {:.3}'.format(E_p_arr[100])+'^{+'+'{:.2}'.format(DE_p_arr_u[100])+'}_{-'+'{:.2}'.format(DE_p_arr_d[100])+'}$, '+'$\\alpha = {:.3}0'.format(alpha_arr[100])+'\\pm{:.2}'.format(Dalpha_arr_u[100])+'}$')
fig.tight_layout()
fig.savefig('./pics_misc/CPL_spectra_ex.pdf')

#Определим космологические модели

def mu_2par_residuals(pars, z_arg, mu_A_arg):
    return np.vectorize(lambda z : mu(z, (70, pars[0], pars[1], pars[0] + pars[1] - 1)))(z_arg) - mu_A_arg

def mu_1par_residuals(pars, z_arg, mu_A_arg):
    return np.vectorize(lambda z : mu(z, (70, pars[0], 1 - pars[0], 0.0)))(z_arg) - mu_A_arg

def loopfun_lcdm(i):
    Omm_2p, OmDE_2p = least_squares(mu_2par_residuals, (0.3, 0.7), loss='soft_l1', f_scale=1.0, bounds=(np.array([0.0, 0.0]), np.array([1.0, 1.0])), args=(z_arr, mu_A_inv_all_samples[:,i])).x
    Omm_1p = least_squares(mu_1par_residuals, (0.3), loss='soft_l1', f_scale=1.0, args=(z_arr, mu_A_inv_all_samples[:,i])).x[0]
    return (Omm_2p, OmDE_2p, Omm_1p)

Omm_2p_sample, OmDE_2p_sample, Omm_1p_sample = np.array(list(zip(*Parallel(n_jobs=Nthreads, max_nbytes='2048M', verbose=10)(delayed(loopfun_lcdm)(i) for i in np.arange(sample_size)))))

fig, ax = plt.subplots()
ax.hist(Omm_1p_sample, bins=40)

def getlegend(pars_cosm):
    H0, Omm, OmDE, Omk = pars_cosm
    return r'$H_0 = '+str(np.around(H0, decimals=1))+r'\,\frac{\mathrm{km/s}}{\mathrm{Mpc}},\; \Omega_{\mathrm{m}} = '+str(np.around(Omm, decimals=3))+r',\; \Omega_{\mathrm{DE}} = '+str(np.around(OmDE, decimals=3))+r',\; \Omega_{\mathrm{k}} = '+str(np.around(Omk, decimals=3))+r'$'

Omm_2p_med = np.median(Omm_2p_sample)
OmDE_2p_med = np.median(OmDE_2p_sample)
Omm_1p_med = np.median(Omm_1p_sample)

mu_arr_2p = np.vectorize(lambda z: mu(z, (70, Omm_2p_med, OmDE_2p_med, Omm_2p_med + OmDE_2p_med - 1)))(z_log_range)
mu_arr_1p = np.vectorize(lambda z: mu(z, (70, Omm_1p_med, 1 - Omm_1p_med, 0.0)))(z_log_range)

fig, ax = plt.subplots()
ax.plot(z_log_range, mu_arr_st, c='k', zorder=10)
#ax.plot(z_log_range, mu_arr_2p, c='tab:orange', zorder=8)
ax.plot(z_log_range, mu_arr_1p, c='green', zorder=11)
ax.errorbar(z_arr, mu_A_inv_meds, yerr=np.array([mu_A_inv_dlim, mu_A_inv_ulim]), linestyle='', linewidth=0.3, marker='o', markersize=1.5, c='teal')
ax.set_xscale('log')
ax.set_ylim((32, 52))
ax.set_title('GRB Hubble Diagram')
ax.set_xlabel(r'$z$')
ax.set_ylabel(r'$\mu$')
fig.tight_layout()
#fig.savefig('pics/HD_icf.png', dpi=300)

qst = np.linspace(0,1,1002)[1:-1]
xst = np.vectorize(smooth_split_normal_ppf)(qst, 5, 1.0, 2.0)
xst2 = np.linspace(xst[0], xst[-1], 1000)
qst2 = np.vectorize(split_normal_cdf)(xst2, 5, 1.0, 2.0)
qst3 = np.vectorize(offset_lognorm_cdf)(xst2, 5, 1.0, 2.0)
fig, ax = plt.subplots()
ax.plot(xst, qst)
ax.plot(xst2, qst2)
ax.plot(xst2, qst3)
ax.plot(np.array([4, 5, 7]), np.array([err_ql, 0.5, err_qu]), linestyle='', marker='o', c='k', markersize=2)

yst2 = 1 / np.gradient(xst, qst)
yst3 = np.gradient(qst2, xst2)
yst4 = np.gradient(qst3, xst2)
fig, ax = plt.subplots()
ax.plot(xst2, yst3, linestyle='dashed', color='darkgray')
#ax.plot(xst2, yst4, color='darkgray')
ax.plot(xst, yst2, color='k')

test = random_smooth_split_normal(5, 2, 1, 10000)
fig, ax = plt.subplots()
ax.hist(test, bins=50)

xst2 = np.linspace(0.16, 0.84, 51)
yst20 = (np.cos((xst2 - 0.5) / (err_qu - 0.5) * pi) + 1) / 2.0
yst21 = np.sqrt(1 - ((xst2 - 0.5) / (err_qu - 0.5)) ** 2)
fig, ax = plt.subplots()
ax.plot(xst2, yst20)
ax.plot(xst2, yst21)