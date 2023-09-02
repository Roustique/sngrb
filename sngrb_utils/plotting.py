import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc, rcParams
import corner
from sngrb_utils import sampling, cosmology

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


def plot_corner(imagename, name_data_dict):
    fig = plt.figure(figsize=(5, 5))
    fig = corner.corner(np.array(list(name_data_dict.values())).T, fig=fig, labels=list(name_data_dict.keys()),
                        quantiles=[sampling.ERR_QL, 0.5, sampling.ERR_QU],
                        show_titles=True, bins=30)
    fig.savefig(f'pics/{imagename}.pdf', dpi=300)


def plot_amati(data, imagename, legendname, plot_errbars=True, plot_samples=True, xlim=(0.75, 4.25),
               ylim=(49, 55), col1='darkgreen', col2='darkgreen', cmap='Greens'):
    """A function for making Amati plane plots"""
    fig, ax = plt.subplots()
    if plot_errbars and plot_samples:
        a_sample, b_sample, Amx_flat, Amy_flat, Amx_meds, Amy_meds, Amx_ulim, Amy_ulim, Amx_dlim, Amy_dlim = data
    elif plot_errbars:
        a_sample, b_sample, Amx_meds, Amy_meds, Amx_ulim, Amy_ulim, Amx_dlim, Amy_dlim = data
    elif plot_samples:
        a_sample, b_sample, Amx_flat, Amy_flat, Amx_meds, Amy_meds = data
    a_est = np.median(a_sample)
    b_est = np.median(b_sample)
    sample_size = len(a_sample)
    Da_est = (np.percentile(a_sample, 100 * sampling.ERR_QU) - np.percentile(a_sample, 100 * sampling.ERR_QL)) / 2
    Db_est = (np.percentile(b_sample, 100 * sampling.ERR_QU) - np.percentile(b_sample, 100 * sampling.ERR_QL)) / 2
    x_range = np.linspace(xlim[0], xlim[1], 100)
    y_range = sampling.lin(x_range, a_est, b_est)
    y_range_1sig_d, y_range_1sig_u = sampling.lin_band(x_range, a_sample, b_sample)
    Am_res_sigma = (Amy_meds - np.median(a_sample) * Amx_meds - np.median(b_sample)).std()
    ax.plot(x_range, y_range, c='k', zorder=10)
    ax.plot([], [], c='grey', linestyle=':', linewidth=0.75)
    ax.plot(x_range, y_range + Am_res_sigma, c='k', zorder=10, linestyle='dashed', linewidth=1)
    ax.fill_between(x_range, y_range_1sig_d, y_range_1sig_u, alpha=0.2, color='k', linestyle=':', linewidth=0.75)
    ax.plot(x_range, y_range - Am_res_sigma, c='k', zorder=10, linestyle='dashed', linewidth=1)
    if plot_errbars:
        ax.errorbar(Amx_meds, Amy_meds, yerr=np.array([Amy_dlim, Amy_ulim]), xerr=np.array([Amx_dlim, Amx_ulim]),
                    linestyle='', linewidth=0.3, marker='o', markersize=1.15, color=col1, rasterized=True)
    if plot_samples:
        limmask = (Amx_flat > xlim[0]) * (Amx_flat < xlim[1]) * (Amy_flat > ylim[0]) * (Amy_flat < ylim[1])
        ax.scatter(Amx_flat[limmask], Amy_flat[limmask], s=0.0015 * (1000 / sample_size) ** 1.5, c=col2, marker=".",
                   rasterized=True)
        # ax.scatter_density(Amx_flat[limmask], Amy_flat[limmask], color=col2, vmin=0, vmax=50)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_title('Amati plane, ' + legendname)
    ax.set_xlabel(r'$\mathrm{log}\,(E_{\mathrm{p,i}} \,/\, 1\,\mathrm{keV})$')
    ax.set_ylabel(r'$\mathrm{log}\,(E_{\mathrm{iso}}\,/\, 1\,\mathrm{erg})$')
    ax.legend(['$a=' + str(np.around(a_est, 2)) + '\\pm' + str(np.around(Da_est, 2)) + '$,\n$b=' + str(
        np.around(b_est, 1)) + '\\pm' + str(np.around(Db_est, 1)) + '$', '$1\\sigma$-conf. region',
               '$1\\sigma$-pred. band'], loc=4)
    fig.tight_layout()
    if plot_errbars and plot_samples:
        fig.savefig('pics/' + imagename + '_errbars_and_samples.pdf', dpi=225)
    elif plot_errbars:
        fig.savefig('pics/' + imagename + '_errbars.pdf', dpi=225)
    elif plot_samples:
        fig.savefig('pics/' + imagename + '_samples.pdf', dpi=225)


def plot_hd(imagename, z_arr, mu_a_meds, mu_a_dlim, mu_a_ulim):
    z_log_range = 10 ** np.linspace(np.min(np.log10(z_arr)), np.max(np.log10(z_arr)), 101)
    mu_arr_st = np.empty(101)
    for i in np.arange(101):
        mu_arr_st[i] = cosmology.mu(z_log_range[i], cosmology.PARS_COSM)

    fig, ax = plt.subplots()
    ax.plot(z_log_range, mu_arr_st, c='k', zorder=10)
    ax.errorbar(z_arr, mu_a_meds, yerr=np.array([mu_a_dlim, mu_a_ulim]), linestyle='', linewidth=0.3, marker='o',
                markersize=1.5, c='teal')
    ax.set_xscale('log')
    ax.set_ylim((32, 52))
    ax.set_title('GRB Hubble Diagram')
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$\mu$')
    fig.tight_layout()
    fig.savefig(f'pics/{imagename}.png', dpi=300)