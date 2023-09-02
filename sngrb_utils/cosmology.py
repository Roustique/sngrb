import numpy as np
from numba import njit, vectorize

pi = 4 * np.arctan(1.0)
c = 299792.458  # km/sec
H0 = 70
PARS_COSM = (H0, 0.3089, 0.6911, 0.0)
MPC_IN_CM = 3.08567758e24

INTIT_DIST = 10001
INTIT_SP = 10001
INTIT_SPERR = 10001


@njit
def cosm_r(z, pars_cosm, am=INTIT_DIST):
    z_int_range = np.linspace(0, z, am)
    h_int_range = 1 / cosm_h(z_int_range, pars_cosm)
    z_int = np.trapz(h_int_range, z_int_range)
    return c / H0 * z_int


@njit
def cosm_h(z, pars_cosm):
    _, om_m, om_de, om_k = pars_cosm
    return np.sqrt(om_m * (1 + z) ** 3 + om_de - om_k * (1 + z) ** 2)


@njit
def cosm_l(z, pars_cosm, am=INTIT_DIST):
    h0, om_m, om_de, om_k = pars_cosm
    z_int_range = np.linspace(0, z, am)
    h_int_range = 1 / cosm_h(z_int_range, pars_cosm)
    z_int = np.trapz(h_int_range, z_int_range)
    if om_k == 0:
        return c / h0 * z_int
    elif om_k > 0:
        return c / h0 / np.sqrt(np.abs(om_k)) * np.sin(np.sqrt(np.abs(om_k)) * z_int)
    else:
        return c / h0 / np.sqrt(np.abs(om_k)) * np.sinh(np.sqrt(np.abs(om_k)) * z_int)


@njit
def dl(z, pars_cosm):
    return (1 + z) * cosm_l(z, pars_cosm)


@njit
def de(z, pars_cosm):
    return np.sqrt(1 + z) * cosm_l(z, pars_cosm)


@njit
def mu(z, pars_cosm):
    return 25 + 5 * np.log10(dl(z, pars_cosm))


@vectorize(['float64(float64)'])
def mu_sn_vec(z):
    return mu(z, PARS_COSM)


@njit
def en(e, pars_spec):
    alpha, e_p = pars_spec
    e_0 = e_p / (2 + alpha)
    return e ** (alpha + 1) * np.exp(-e / e_0)


@njit
def int_e(lims, pars_spec, am=INTIT_SP):
    e_range = np.linspace(lims[0], lims[1], am)
    en_range = en(e_range, pars_spec)
    return np.trapz(en_range, e_range)


@njit
def s_bolo(z, alpha, e_p, s_obs):
    denom_int = int_e((15, 150), (alpha, e_p))
    num_int = int_e((1 / (1 + z), 1e4 / (1 + z)), (alpha, e_p))
    if denom_int == 0:
        return np.inf
    else:
        return s_obs * num_int / denom_int * 1e-7


@njit
def e_iso(z, alpha, e_p, s_obs, pars_cosm):
    return 4 * pi * (de(z, pars_cosm) * MPC_IN_CM) ** 2 * s_bolo(z, alpha, e_p, s_obs)


@njit
def mu_a(z, alpha, e_p, s_obs, a, b, k_par):
    s_bolo_arg = s_bolo(z, alpha, e_p, s_obs)
    if s_bolo_arg == 0 or s_bolo_arg > 1e10:
        return 0
    else:
        return 25 + 2.5 * (np.log10((1 + z) ** (k_par + 1) / (4 * pi * s_bolo_arg * MPC_IN_CM ** 2)) + a * np.log10(
            e_p * (1 + z)) + b)


@vectorize(['float64(float64,float64,float64,float64,float64,float64,float64)'])
def mu_a_vec(z, alpha, e_p, s_obs, a, b, k_par):
    return mu_a(z, alpha, e_p, s_obs, a, b, k_par)


@njit
def amx(z, e_p):
    return np.log10(e_p * (1 + z))


@njit
def amy(z, alpha, e_p, s_obs, pars_cosm):
    return np.log10(e_iso(z, alpha, e_p, s_obs, pars_cosm))
