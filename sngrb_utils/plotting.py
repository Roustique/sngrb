import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rc, rcParams
import corner
import mpl_scatter_density
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from sngrb_utils import sampling, cosmology

imnorm = ImageNormalize(vmin=2, vmax=1000, stretch=LogStretch())

paper_linewidth = 3.37689
paper_textwidth = 7.03058

JOURNAL = 'MNRAS'

rc('font', family='Times New Roman')
rc('font', size=10)
rc('mathtext', fontset='stix')
if JOURNAL == 'Universe':
    rc('font', family='tex gyre pagella')
    rcParams['font.sans-serif'] = 'tex gyre pagella'
    rc('font', size=12)
    rc('mathtext', fontset='custom', it='tex gyre pagella:italic')

rc('text', usetex=False)
rc('text.latex', preamble=r'\usepackage[utf8]{inputenc}')
rc('text.latex', preamble=r'\usepackage[russian]{babel}')
rc('figure', figsize=(5, 4.25))

cat_sn = pd.read_csv('catalogues/Pantheon.dat', delimiter='\t', header=0, usecols=[2, 4, 5])
z_arr_sn = np.array(cat_sn['zcmb'])
mu_arr_sn = np.array(cat_sn['mu'])  # + 19.41623729
d_mu_arr_sn = np.array(cat_sn['err_mu'])
z_log_range = 10 ** np.linspace(np.log10(np.min(z_arr_sn)), np.log10(np.max(z_arr_sn)))


def plot_corner(imagename, name_data_dict):
    fig = plt.figure(figsize=(5, 5))
    fig = corner.corner(np.array(list(name_data_dict.values())).T, fig=fig, labels=list(name_data_dict.keys()),
                        quantiles=[sampling.ERR_QL, 0.5, sampling.ERR_QU],
                        show_titles=True, bins=30)
    fig.savefig(f'pics/{imagename}.pdf', dpi=300)


def plot_amati(
        data,
        imagename,
        legendname,
        plot_errbars=True,
        plot_samples=True,
        xlim=(0.75, 4.25),
        ylim=(49, 55),
        col1='indigo',
        col2='indigo',
        cmap='PuBu'
):
    """A function for making Amati plane plots"""
    fig = plt.figure(figsize=[6, 5])
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    if plot_errbars and not plot_samples:
        a_sample, b_sample, amx_meds, amy_meds, amx_ulim, amy_ulim, amx_dlim, amy_dlim = data
    elif plot_samples and not plot_errbars:
        a_sample, b_sample, amx_flat, amy_flat, amx_meds, amy_meds = data
    else:
        a_sample, b_sample, amx_flat, amy_flat, amx_meds, amy_meds, amx_ulim, amy_ulim, amx_dlim, amy_dlim = data
    a_est = np.median(a_sample)
    b_est = np.median(b_sample)
    d_a_est = (np.percentile(a_sample, 100 * sampling.ERR_QU) - np.percentile(a_sample, 100 * sampling.ERR_QL)) / 2
    d_b_est = (np.percentile(b_sample, 100 * sampling.ERR_QU) - np.percentile(b_sample, 100 * sampling.ERR_QL)) / 2
    x_range = np.linspace(xlim[0], xlim[1], 100)
    y_range = sampling.lin(x_range, a_est, b_est)
    y_range_1sig_d, y_range_1sig_u = sampling.lin_band(x_range, a_sample, b_sample)
    am_res_sigma = (amy_meds - np.median(a_sample) * amx_meds - np.median(b_sample)).std()
    ax.plot(x_range, y_range, c='k', zorder=10)
    ax.plot([], [], c='grey', linestyle=':', linewidth=0.75)
    ax.plot(x_range, y_range + am_res_sigma, c='k', zorder=10, linestyle='dashed', linewidth=1)
    ax.fill_between(x_range, y_range_1sig_d, y_range_1sig_u, alpha=0.2, color='k', linestyle=':', linewidth=0.75)
    ax.plot(x_range, y_range - am_res_sigma, c='k', zorder=10, linestyle='dashed', linewidth=1)
    if plot_errbars:
        ax.errorbar(amx_meds, amy_meds, yerr=np.array([amy_dlim, amy_ulim]), xerr=np.array([amx_dlim, amx_ulim]),
                    linestyle='', linewidth=0.3, marker='o', markersize=1.15, color=col1, rasterized=True)
    if plot_samples:
        limmask = (amx_flat > xlim[0]) * (amx_flat < xlim[1]) * (amy_flat > ylim[0]) * (amy_flat < ylim[1])
        ax.scatter_density(amx_flat[limmask], amy_flat[limmask], cmap='BuPu', norm=imnorm)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_title(legendname)
    ax.set_xlabel(r'$\mathrm{log}\,(E_{\mathrm{p,i}} \,/\, 1\,\mathrm{keV})$')
    ax.set_ylabel(r'$\mathrm{log}\,(E_{\mathrm{iso}}\,/\, 1\,\mathrm{erg})$')
    ax.legend(['$a=' + str(np.around(a_est, 2)) + '\\pm' + str(np.around(d_a_est, 2)) + '$,\n$b=' + str(
        np.around(b_est, 1)) + '\\pm' + str(np.around(d_b_est, 1)) + '$', '$1\\sigma$-conf. band',
               '$1\\sigma$-pred. band'], loc=4)
    fig.tight_layout()
    if plot_errbars and plot_samples:
        fig.savefig('pics/' + imagename + '_errbars_and_samples.pdf', dpi=225)
        fig.savefig('pics/' + imagename + '_errbars_and_samples.png', dpi=225)
    elif plot_errbars:
        fig.savefig('pics/' + imagename + '_errbars.pdf', dpi=225)
        fig.savefig('pics/' + imagename + '_errbars.png', dpi=225)
    elif plot_samples:
        fig.savefig('pics/' + imagename + '_samples.pdf', dpi=225)
        fig.savefig('pics/' + imagename + '_samples.png', dpi=225)


def plot_hd(imagename, problemname, z_arr, mu_a_meds, mu_a_dlim, mu_a_ulim):
    z_log_range_big = 10 ** np.linspace(np.log10(np.min(z_arr_sn)), np.log10(np.max(z_arr)))
    mu_range_cosm = cosmology.mu_cosm_vec(z_log_range_big)
    delta_mu_arr_sn = mu_arr_sn - cosmology.mu_cosm_vec(z_arr_sn)
    delta_mu_a_meds = mu_a_meds - cosmology.mu_cosm_vec(z_arr)

    fig, (ax, ax_low) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3.5, 1]}, figsize=(7, 5.5))
    ax.errorbar(z_arr, mu_a_meds, yerr=np.array([mu_a_dlim, mu_a_ulim]), linestyle='', markeredgewidth=0.25,
                markeredgecolor='k', linewidth=0.3, marker='o', markersize=3, c='teal', zorder=4, rasterized=True)
    ax.errorbar(z_arr_sn, mu_arr_sn, d_mu_arr_sn, color='silver', linestyle='', marker='o', markeredgewidth=0.25,
                markeredgecolor='k', markersize=2.5, rasterized=True)
    ax.plot(z_log_range_big, mu_range_cosm, c='k', zorder=10)
    ax_low.errorbar(z_arr, delta_mu_a_meds, yerr=np.array([mu_a_dlim, mu_a_ulim]), linestyle='', markeredgewidth=0.25,
                    markeredgecolor='k', linewidth=0.3, marker='o', markersize=3, c='teal', zorder=4, rasterized=True)
    ax_low.errorbar(z_arr_sn, delta_mu_arr_sn, d_mu_arr_sn, color='silver', linestyle='', marker='o',
                    markeredgewidth=0.25, markeredgecolor='k', markersize=2.5, rasterized=True)
    ax_low.plot(z_log_range_big, np.zeros(z_log_range_big.size), c='k', zorder=10)
    ax.legend(['$\Lambda$CDM', 'GRB', 'SN Ia (Pantheon)'], loc='upper left', fancybox=True, shadow=True)
    ax.set_xscale('log')
    ax_low.set_xscale('log')
    ax.set_ylim((32, 53))
    ax_low.set_ylim((-7.5, 7.5))
    ax.set_title(f'GRB+SN Hubble Diagram ({problemname})')
    ax.set_ylabel(r'Distance modulus $\mu$')
    ax_low.set_ylabel(r'$\Delta\mu$')
    ax_low.set_xlabel(r'Redshift $z$')
    fig.tight_layout()
    fig.savefig(f'pics/{imagename}.png', dpi=300)
    fig.savefig(f'pics/{imagename}.pdf', dpi=300)
