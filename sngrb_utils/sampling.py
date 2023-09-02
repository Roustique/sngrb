import numpy as np
from scipy.stats import norm
import cosmology

ERR_QL = norm.cdf(-1.0)
ERR_QU = norm.cdf(1.0)


def lin(x, a, b):
    """Simple linear function"""
    return a * x + b


def lin_band(
        xs,
        a_sample,
        b_sample,
        conf_int=100 * (ERR_QU - ERR_QL)
):
    """Function which returns upper and lower sigma-bands for a linear function"""
    upper_bord = np.vectorize(lambda x: np.percentile(lin(x, a_sample, b_sample), (100 + conf_int) / 2))(xs)
    lower_bord = np.vectorize(lambda x: np.percentile(lin(x, a_sample, b_sample), (100 - conf_int) / 2))(xs)
    return lower_bord, upper_bord


def lin_inv(y, a, b):
    """Linear function with swapped x and y"""
    return (y - b) / a


def split_normal_cdf(
        x: float,
        mu: float,
        lower_sigma: float,
        upper_sigma: float
):
    """A cumulative distribution function of a split-normal distribution"""
    if x < mu:
        return norm.cdf(x, mu, lower_sigma)
    else:
        return norm.cdf(x, mu, upper_sigma)


def random_split_normal(
        mu: float,
        lower_sigma: float,
        upper_sigma: float,
        amount: int = 1024
):
    """A function for sampling from the split-normal distribution"""
    z = np.random.normal(0, 1, amount)
    return mu + z * (- lower_sigma * (np.sign(z) - 1) / 2 + upper_sigma * (np.sign(z) + 1) / 2)


def random_log_split_normal(
        mu: float,
        lower_sigma: float,
        upper_sigma: float,
        amount: int = 1024
):
    """A function for sampling from the log-split-normal distribution"""
    return 10 ** (
        random_split_normal(np.log10(mu), np.log10(mu / (mu - lower_sigma)), np.log10((mu + upper_sigma) / mu),
                            amount=amount))


def offset_lognorm_ppf(q, mu, lower_sigma, upper_sigma):
    """A percentile function of the shifted log-normal distribution (unused)"""
    if lower_sigma < upper_sigma:
        delta = mu - lower_sigma * upper_sigma / (upper_sigma - lower_sigma)
        return 10 ** norm.ppf(q, np.log10(mu - delta), np.log10((mu + upper_sigma - delta) / (mu - delta))) + delta
    elif lower_sigma > upper_sigma:
        delta = mu - lower_sigma * upper_sigma / (lower_sigma - upper_sigma)
        return - 10 ** norm.ppf(1 - q, np.log10(mu - delta),
                                np.log10((mu + lower_sigma - delta) / (mu - delta))) - delta + 2 * mu
    else:
        return norm.ppf(q, mu, upper_sigma)


def smooth_split_normal_ppf(q, mu, lower_sigma, upper_sigma):
    """A percentile function of the smoothed split-normal distribution (unused)"""
    if (q > norm.cdf(-2.0)) and (q < 0.5):
        x = (q - 0.5) / (0.5 - norm.cdf(-2.0))
        w = (np.cos(x * cosmology.pi) + 1) / 2.0
        return offset_lognorm_ppf(q, mu, lower_sigma, upper_sigma) * w + norm.ppf(q, mu, lower_sigma) * (1 - w)
    elif (q >= 0.5) and (q < norm.cdf(2.0)):
        x = (q - 0.5) / (norm.cdf(2.0) - 0.5)
        w = (np.cos(x * cosmology.pi) + 1) / 2.0
        return offset_lognorm_ppf(q, mu, lower_sigma, upper_sigma) * w + norm.ppf(q, mu, upper_sigma) * (1 - w)
    elif q <= norm.cdf(-2.0):
        return norm.ppf(q, mu, lower_sigma)
    else:
        return norm.ppf(q, mu, upper_sigma)


def random_smooth_split_normal(mu, lower_sigma, upper_sigma, amount=1024):
    """A function for sampling from the smoothed split-normal distribution (unused)"""
    return np.vectorize(smooth_split_normal_ppf)(np.random.uniform(size=amount), mu, lower_sigma, upper_sigma)


def make_samples(ii, data, amount=1024):
    """A function for sampling alpha, E_p and S_obs from the split-normal distribution"""
    alpha_arg, d_alpha_arg_d, d_alpha_arg_u, e_p_arg, d_e_p_arg_d, d_e_p_arg_u, s_obs_arg, d_s_obs_arg, z_arg = data
    alpha_sample = random_split_normal(alpha_arg[ii], d_alpha_arg_d[ii], d_alpha_arg_u[ii], amount)
    e_p_sample = random_log_split_normal(e_p_arg[ii], d_e_p_arg_d[ii], d_e_p_arg_u[ii], amount=amount)
    s_obs_sample = np.random.normal(s_obs_arg[ii], d_s_obs_arg[ii], amount)
    return alpha_sample, e_p_sample, s_obs_sample


def get_meds_and_lims(samples):
    """A function that returns the median and the credible interval limits for a given sample"""
    n_inner = np.shape(samples)[0]
    meds = np.empty(n_inner)
    dlim = np.empty(n_inner)
    ulim = np.empty(n_inner)
    for i in np.arange(n_inner):
        meds[i] = np.median(samples[i, :][np.isfinite(samples[i, :])])
        dlim[i] = meds[i] - np.percentile(samples[i, :][np.isfinite(samples[i, :])], 100 * ERR_QL)
        ulim[i] = np.percentile(samples[i, :][np.isfinite(samples[i, :])], 100 * ERR_QU) - meds[i]
    return meds, dlim, ulim


def sample_amatixy(ii, data):
    """A function for calculating Amati x and y for samples of arguments"""
    alpha_sample, e_p_sample, s_obs_sample, z_arg = data
    amx_sample = cosmology.amx(z_arg[ii], e_p_sample)
    amx_sample[np.isnan(amx_sample)] = -np.inf
    amy_sample = np.vectorize(
        lambda z, alpha, e_p, s_obs: cosmology.amy(z, alpha, e_p, s_obs, cosmology.PARS_COSM)
    )(
            z_arg[ii],
            alpha_sample,
            e_p_sample,
            s_obs_sample
    )
    amy_sample[np.isnan(amy_sample)] = np.inf
    return amx_sample, amy_sample


def sample_s_bolo(ii, data):
    """A function for calculating the S_bolo for samples of arguments"""
    alpha_sample, e_p_sample, s_obs_sample, z_arg = data
    s_bolo_sample = cosmology.s_bolo(z_arg[ii], alpha_sample, e_p_sample, s_obs_sample) * cosmology.MPC_IN_CM ** 2
    return s_bolo_sample


def sample_mu_a(ii, data, a, b, k):
    """A function for calculating the mu_A for samples of arguments"""
    alpha_sample, e_p_sample, s_obs_sample, z_arg = data
    mu_a_sample = np.vectorize(cosmology.mu_a)(z_arg[ii], alpha_sample, e_p_sample, s_obs_sample, a, b, k)
    return mu_a_sample
