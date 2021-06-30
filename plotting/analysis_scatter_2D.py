"""Scatter plot for the nRMSEa against KS entropy and the first Lyapunov exponent using full, frequent observations.
"""

import matplotlib.pyplot as plt
import matplotlib.markers as mrks
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import scipy.stats.mstats
import os
import subprocess

def plot_scatter_KS(d_RMSE, d_KS, d_LE, d_n0, dt):

    fig = plt.figure(1, figsize = (12, 8))
    figlegend = plt.figure(2, figsize = (32, 6))
    fig.clf()
    ax = fig.add_subplot(111)

    axins = inset_axes(ax, width="40%", height="40%", loc = 4, borderpad=2.8)

    markers = {}
    markers[f'F10G0'] = mrks.MarkerStyle('o', 'none')
    markers[f'F10G10'] = mrks.MarkerStyle('p', 'none')
    markers[f'F0G10'] = mrks.MarkerStyle('x', 'none')

    for F, G in zip([10, 10, 0], [10, 0, 10]):
        n0 = d_n0[f'F{F}G{G}']

        # plot the main figure
        KS = np.where(d_KS[f'F{F}G{G}'] < 1., np.nan, d_KS[f'F{F}G{G}'])
        RMSE = np.where(KS == np.nan, np.nan, d_RMSE[f'F{F}G{G}'])
        upperbound = np.where((KS != np.nan) & (d_LE[f'F{F}G{G}'] > 0), np.sqrt(n0*(np.exp(2*d_LE[f'F{F}G{G}']*dt) - 1)/36), np.nan)

        ax.scatter(np.log(RMSE), np.log(KS), marker = markers[f'F{F}G{G}'], color ='k', label = rf'nRMSEa (F={F}, G={G})', s = 120)
        ax.scatter(np.log(upperbound), np.log(KS), marker = markers[f'F{F}G{G}'], color = 'r', label=rf"upper bound (F={F}, G={G})", s = 120)

        # plot the inset
        KS = np.where((d_KS[f'F{F}G{G}'] >= 1.) | (d_KS[f'F{F}G{G}'] <= 1e-6), np.nan, d_KS[f'F{F}G{G}'])
        RMSE = np.where(KS == np.nan, np.nan, d_RMSE[f'F{F}G{G}'])
        upperbound = np.where((KS != np.nan) & (d_LE[f'F{F}G{G}'] > 0), np.sqrt(n0*(np.exp(2*d_LE[f'F{F}G{G}']*dt) - 1)/36), np.nan)

        axins.scatter(np.log(RMSE), np.log(KS), marker = markers[f'F{F}G{G}'], color ='k', label = rf'nRMSEa (F={F}, G={G})', s = 120)
        axins.scatter(np.log(upperbound), np.log(KS), marker = markers[f'F{F}G{G}'], color = 'r', label=rf"upper bound (F={F}, G={G})", s = 120)

    # plt.axhline(-20., -2.5)
    # KS = np.concatenate(list(d_KS.values())).flatten()
    # RMSE = np.concatenate(list(d_RMSE.values())).flatten()
    # RMSE = np.log(RMSE)
    # KS = np.ma.masked_where((KS <= 1) | (RMSE == np.nan) | (RMSE == np.inf), np.log(KS))
    # RMSE = np.ma.masked_where(KS.mask, RMSE)
    # print(np.isnan(RMSE).sum(), np.isnan(KS).sum(), np.isinf(RMSE).sum(),np.isinf(KS).sum())
    # res = scipy.stats.mstats.linregress(RMSE, y=KS)
    # print(res.slope, res.intercept)
    #(1.775403148086777 5.368677073889142)
    # plt.plot(np.arange(-3., -1., 0.01), res.slope*np.arange(-3., -1., 0.01) + res.intercept, color = 'r')
    lgnd = figlegend.legend(*ax.get_legend_handles_labels(), 'center', prop = {'size': 24}, ncol=3, handlelength=2.)

    axins.set_ylim([-12, 1])
    axins.set_yticks(np.arange(-12, 2, 3))
    axins.set_yticklabels(np.round(np.arange(-12, 2, 3), 2), fontsize=28)
    axins.set_xlim([-12, -2.])
    axins.set_xticks(np.arange(-12, 0, 3))
    axins.set_xticklabels(np.round(np.arange(-12, 0, 3), 2), fontsize=28)

    ax.set_xlabel(r'ln(nRMSEa)', fontsize=32)
    ax.set_ylabel(r'ln($\sigma_{KS}$)', fontsize=32)
    ax.set_ylim([-0.1, 3.5])
    ax.set_yticks(np.arange(0, 4, 1))
    ax.set_yticklabels(np.round(np.arange(0, 4, 1), 2), fontsize=28)
    ax.set_xlim([-3.5, 0.5])
    ax.set_xticks(np.arange(-3.5, 1, 0.5))
    ax.set_xticklabels(np.round(np.arange(-3.5, 1, 0.5), 2), fontsize=28)
    
    figlegend.savefig('legend.pdf')
    figname = "RMSE_KE_relation_scatter_loglog"
    fig.savefig(figname+'.pdf')
    subprocess.run(['pdfcrop', figname+'.pdf'])
    os.remove(figname+'.pdf')
    os.rename(figname+'-crop.pdf',figname+'.pdf')

    subprocess.run(['pdfcrop', 'legend.pdf'])
    os.remove('legend.pdf')
    os.rename('legend-crop.pdf','legend.pdf')


def plot_scatter_LE1(d_RMSE, d_KS, d_LE, d_n0, dt):

    fig = plt.figure(3, figsize = (12, 8))
    fig.clf()

    ax = fig.add_subplot(111)

    axins = inset_axes(ax, width="40%", height="40%", loc = 4, borderpad=2.8)

    markers = {}
    markers[f'F10G0'] = mrks.MarkerStyle('o', 'none')
    markers[f'F10G10'] = mrks.MarkerStyle('p', 'none')
    markers[f'F0G10'] = mrks.MarkerStyle('x', 'none')

    for F, G in zip([10, 10, 0], [10, 0, 10]):
        LE = np.where(d_KS[f'F{F}G{G}'] < 1, np.nan, d_LE[f'F{F}G{G}'])
        RMSE = np.where(LE == np.nan, np.nan, d_RMSE[f'F{F}G{G}'])
        upperbound = np.where((LE != np.nan) & (d_LE[f'F{F}G{G}'] > 0), np.sqrt(d_n0[f'F{F}G{G}']*(np.exp(2*d_LE[f'F{F}G{G}']*dt) - 1)/36), np.nan)

        ax.scatter(np.log(RMSE), np.log(LE), marker = markers[f'F{F}G{G}'], color ='k', label = rf'F={F}, G={G}', s = 120)
        ax.scatter(np.log(upperbound), np.log(LE), marker = markers[f'F{F}G{G}'], color ='r', label="upper bound", s = 120)

        LE = np.where((d_KS[f'F{F}G{G}'] >= 1.) | (d_KS[f'F{F}G{G}'] <= 1e-6), np.nan, d_LE[f'F{F}G{G}'])
        RMSE = np.where(LE == np.nan, np.nan, d_RMSE[f'F{F}G{G}'])
        upperbound = np.where((LE != np.nan) & (d_LE[f'F{F}G{G}'] > 0), np.sqrt(d_n0[f'F{F}G{G}']*(np.exp(2*d_LE[f'F{F}G{G}']*0.05) - 1)/36), np.nan)
        axins.scatter(np.log(RMSE), np.log(LE), marker = markers[f'F{F}G{G}'], color ='k', label = rf'F={F}, G={G}', s = 120)
        axins.scatter(np.log(upperbound), np.log(LE), marker = markers[f'F{F}G{G}'], color ='r', label="upper bound", s = 120)

    axins.set_ylim([-12, 1])
    axins.set_yticks(np.arange(-12, 2, 3))
    axins.set_yticklabels(np.round(np.arange(-12, 2, 3), 2), fontsize=28)
    axins.set_xlim([-12, -2.])
    axins.set_xticks(np.arange(-12, 0, 3))
    axins.set_xticklabels(np.round(np.arange(-12, 0, 3), 2), fontsize=28)

    ax.set_xlabel(r'ln(nRMSEa)', fontsize=32)
    ax.set_ylabel(r'ln($\lambda_1$)', fontsize=32)
    ax.set_ylim([-1.2, 1.65])
    ax.set_yticks(np.arange(-1, 2, 1))
    ax.set_yticklabels(np.round(np.arange(-1, 2, 1), 2), fontsize=28)
    ax.set_xlim([-3.5, 0.5])
    ax.set_xticks(np.arange(-3.5, 1, 0.5))
    ax.set_xticklabels(np.round(np.arange(-3.5, 1, 0.5), 2), fontsize=28)

    figname = 'RMSE_LE_relation_scatter_loglog'
    fig.savefig(figname+'.pdf')

    subprocess.run(['pdfcrop', figname+'.pdf'])
    os.remove(figname+'.pdf')
    os.rename(figname+'-crop.pdf',figname+'.pdf')


def analysis(alphas=np.arange(0.1, 3., 0.1), gammas=np.arange(0.3, 1.8, 0.05)):
    times = 5
    nt = 3000
    dt = 0.05

    d_RMSE = {}
    d_KS = {}
    d_LE1 = {}
    d_n0 = {}

    for F, G in zip([10, 10, 0], [10, 0, 10]):
        RMSE = np.zeros([len(alphas), len(gammas)])
        KS = np.zeros([len(alphas), len(gammas)])
        LE1 = np.zeros([len(alphas), len(gammas)])
        n0 = np.zeros([len(alphas), len(gammas)], dtype=int)
        for i_a, alpha in enumerate(alphas):
            for i_g, gamma in enumerate(gammas):
                # read the Lyapunov exponents
                filename = f'../LEs/LE_F{F}_G{G}_alpha{np.round(alpha, 3)}_gamma{np.round(gamma, 3)}.npy'
                LE = np.load(filename)
                n0[i_a, i_g] = np.sum(LE>0)

                # calculate the neutral mode
                if n0[i_a, i_g] > 0:
                    n0[i_a, i_g] += (abs(LE[n0[i_a, i_g]]) < abs(LE[n0[i_a, i_g]-1]))

                # KS entropy
                KS[i_a, i_g] = np.sum(LE[LE>0])
                # first Lyapunov exponent
                LE1[i_a, i_g] = LE[0]

                if KS[i_a, i_g] <= 1e-6:
                    RMSE[i_a, i_g] = np.nan
                    continue

                filename = f'../std/std_F{F}_G{G}_alpha{np.round(alpha, 3)}_gamma{np.round(gamma, 3)}.npy'
                std = np.load(filename)

                if abs(std[1]) <= 1e-6: std[1] = 0.01
                if abs(std[0]) <= 1e-6: std[0] = 0.01
                sigma = np.array([0.05*(std[0]**2), 0.05*(std[1]**2)])

                # get nRMSEa
                filename = f'Nx18_F{F}_G{G}_N40_alpha{np.round(alpha, 3)}_gamma{np.round(gamma, 3)}_all.npz'
                stats = np.load('../RMSE/' + filename)

                x = np.zeros([times])
                for it in range(times):
                    xtmp = 0.5*(stats['rmse_x_a'+str(it)][-nt:]**2/sigma[0] + stats['rmse_theta_a'+str(it)][-nt:]**2/sigma[1])
                    if any(np.isnan(xtmp)):
                        print(F, G, alpha, gamma)
                        xtmp = np.nan
                    x[it] = np.mean(np.sqrt(xtmp))
                RMSE[i_a, i_g] = np.mean(x)

        d_RMSE[f'F{F}G{G}'] = RMSE
        d_KS[f'F{F}G{G}'] = KS
        d_LE1[f'F{F}G{G}'] = LE1
        d_n0[f'F{F}G{G}'] = n0

    # plotting
    plot_scatter_KS(d_RMSE, d_KS, d_LE1, d_n0, dt)
    plot_scatter_LE1(d_RMSE, d_KS, d_LE1, d_n0, dt)

analysis()