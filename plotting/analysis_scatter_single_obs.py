"""Scatter plot with one variable observed with KS entropy and lambda_1.
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats.mstats
import os
import subprocess
import matplotlib.markers as mrks


def plot_scatter_KS(d_RMSE, d_LE, d_KS, d_n0):
    fig = plt.figure(1, figsize = (12, 8))
    figlegend = plt.figure(4, figsize = (32, 6))
    fig.clf()
    ax = fig.add_subplot(111)

    markers = {}
    markers[f'F10G0'] = mrks.MarkerStyle('o', 'none')
    markers[f'F10G10'] = mrks.MarkerStyle('p', 'none')
    markers[f'F0G10'] = mrks.MarkerStyle('x', 'none')

    for F, G in zip([10, 10, 0], [10, 0, 10]):
        KS = np.array([])
        RMSE = np.array([])
        KS_up = np.array([])
        upperbound = np.array([])

        for vary in ['alpha', 'gamma']:
            KS_tmp = np.where(d_KS[f'F{F}G{G}vary{vary}'] < 1., np.nan, d_KS[f'F{F}G{G}vary{vary}'])
            KS_up = np.append(KS_up, KS_tmp.flatten())
            print(d_KS[f'F{F}G{G}vary{vary}'])
            upperbound = np.append(upperbound, 
                np.where(KS_tmp != np.nan, np.sqrt(d_n0[f'F{F}G{G}vary{vary}']*(np.exp(2*d_LE[f'F{F}G{G}vary{vary}']*0.05) - 1)/36), np.nan))

            for i_o, obs_type in enumerate(['X', 'theta']):
                KS = np.append(KS, KS_tmp.flatten())
                RMSE = np.append(RMSE, np.where(KS_tmp == np.nan, np.nan, d_RMSE[f'F{F}G{G}obs{obs_type}vary{vary}']).flatten())

        ax.scatter(np.log(RMSE), np.log(KS), marker = markers[f'F{F}G{G}'], color ='k', label = rf'nRMSEa (F={F}, G={G})', s = 240)
        ax.scatter(np.log(upperbound), np.log(KS_up), marker = markers[f'F{F}G{G}'], color ='r', label=rf"upper bound (F={F}, G={G})", s = 240)

    lgnd = figlegend.legend(*ax.get_legend_handles_labels(), 'center', prop = {'size': 24}, ncol=3, handlelength=2.)
    ax.set_xlabel('ln(nRMSEa)', fontsize=32)
    ax.set_ylabel(r'ln($\sigma_{KS}$)', fontsize=32)
    ax.set_ylim([-0.1, 3.2])
    ax.set_yticks(np.arange(0., 3.5, 0.5))
    ax.set_yticklabels(np.round(np.arange(0., 3.5, 0.5), 2), fontsize=28)
    
    ax.set_xticks(np.arange(-2.5, 0., 0.5))
    ax.set_xticklabels(np.round(np.arange(-2.5, 0., 0.5), 2), fontsize=28)
    ax.set_xlim([-2.9, -0.3])
    # plt.legend(fontsize=28)
    
    figlegend.savefig('legend_single.pdf')
    figname = "single_KE_relation_scatter_loglog"
    fig.savefig(figname+'.pdf')
    subprocess.run(['pdfcrop', figname+'.pdf'])
    os.remove(figname+'.pdf')
    os.rename(figname+'-crop.pdf',figname+'.pdf')

    subprocess.run(['pdfcrop', 'legend_single.pdf'])
    os.remove('legend_single.pdf')
    os.rename('legend_single-crop.pdf','legend_single.pdf')

def plot_scatter_LE1(d_RMSE, d_LE, d_KS, d_n0):
    fig = plt.figure(1, figsize = (12, 8))
    fig.clf()
    ax = fig.add_subplot(111)

    markers = {}
    markers[f'F10G0'] = mrks.MarkerStyle('o', 'none')
    markers[f'F10G10'] = mrks.MarkerStyle('p', 'none')
    markers[f'F0G10'] = mrks.MarkerStyle('x', 'none')

    for F, G in zip([10, 10, 0], [10, 0, 10]):
        LE = np.array([])
        RMSE = np.array([])
        LE_up = np.array([])
        upperbound = np.array([])
        for vary in ['alpha', 'gamma']:
            LE_tmp = np.where(d_KS[f'F{F}G{G}vary{vary}'] < 1., np.nan, d_LE[f'F{F}G{G}vary{vary}'])
            upperbound = np.append(upperbound, 
                np.where(LE_tmp != np.nan, np.sqrt(d_n0[f'F{F}G{G}vary{vary}']*(np.exp(2*d_LE[f'F{F}G{G}vary{vary}']*0.05) - 1)/36), np.nan))
            LE_up = np.append(LE_up, LE_tmp)

            for i_o, obs_type in enumerate(['X', 'theta']):
                
                LE = np.append(LE, LE_tmp.flatten())

                RMSE = np.append(RMSE, np.where(LE_tmp == np.nan, np.nan, d_RMSE[f'F{F}G{G}obs{obs_type}vary{vary}']).flatten())

        ax.scatter(np.log(RMSE), np.log(LE), marker = markers[f'F{F}G{G}'], color ='k', label = rf'nRMSEa (F={F}, G={G})', s = 240)

        ax.scatter(np.log(upperbound), np.log(LE_up), marker = markers[f'F{F}G{G}'], color ='r', label=rf"upper bound (F={F}, G={G})", s = 240)

    ax.set_xlabel('ln(nRMSEa)', fontsize=32)
    ax.set_ylabel(r'ln($\lambda_1$)', fontsize=32)
    ax.set_ylim([-1, 1.5])
    ax.set_yticks(np.arange(-1, 2, 0.5))
    ax.set_yticklabels(np.round(np.arange(-1, 2, 0.5), 2), fontsize=28)
    
    ax.set_xticks(np.arange(-2.5, 0., 0.5))
    ax.set_xticklabels(np.round(np.arange(-2.5, 0., 0.5), 2), fontsize=28)
    ax.set_xlim([-2.9, -0.3])
    # plt.legend(fontsize=28)
    figname = 'single_LE_relation_scatter_loglog'
    fig.savefig(figname+'.pdf')

    subprocess.run(['pdfcrop', figname+'.pdf'])
    os.remove(figname+'.pdf')
    os.rename(figname+'-crop.pdf',figname+'.pdf')

def analysis(alphas=np.arange(0.1, 3., 0.1), gammas=np.arange(0.3, 1.8, 0.05)):

    times = 5
    nt = 3000

    d_RMSE = {}
    d_KS = {}
    d_LE1 = {}
    d_n0 = {}
    
    for vary in ['alpha', 'gamma']:


        for i_o, obs_type in enumerate(['X', 'theta']):
            for F, G in zip([10, 10, 0], [10, 0, 10]):
                if vary == 'alpha':
                    size = len(alphas)
                    iterator = zip(alphas, np.ones(len(alphas)))
                else:
                    size = len(gammas)
                    iterator = zip(np.ones(len(gammas)), gammas)

                RMSE = np.zeros(size)
                KS = np.zeros(size)
                LE1 = np.zeros(size)
                n0 = np.zeros(size, dtype=int)

                for i_a, (alpha, gamma) in enumerate(iterator):

                    filename = f'../LEs/LE_F{F}_G{G}_alpha{np.round(alpha, 3)}_gamma{np.round(gamma, 3)}.npy'
                    LE = np.load(filename)

                    KS[i_a] = np.sum(LE[LE>0])

                    LE1[i_a] = LE[0]

                    n0[i_a] = np.sum(LE>0)
                    if n0[i_a] > 0:
                        n0[i_a] += (abs(LE[n0[i_a]]) < abs(LE[n0[i_a]-1]))

                    if KS[i_a] <= 1e-6:
                        RMSE[i_a] = np.nan
                        continue

                    filename = f'../std/std_F{F}_G{G}_alpha{np.round(alpha, 3)}_gamma{np.round(gamma, 3)}.npy'
                    std = np.load(filename)

                    if abs(std[1]) <= 1e-6: std[1] = 0.01
                    if abs(std[0]) <= 1e-6: std[0] = 0.01
                    sigma = np.array([0.05*(std[0]**2), 0.05*(std[1]**2)])

                    filename = f'Nx18_F{F}_G{G}_N40_alpha{np.round(alpha, 3)}_gamma{np.round(gamma, 3)}_{obs_type}.npz'
                    stats = np.load('../RMSE/' + filename)

                    x = np.zeros([times])
                    for it in range(times):
                        xtmp = 0.5*(stats['rmse_x_a'+str(it)][-nt:]**2/sigma[0] + stats['rmse_theta_a'+str(it)][-nt:]**2/sigma[1])
                        if any(np.isnan(xtmp)):
                            print(F, G, alpha, gamma)
                            xtmp = np.nan
                        x[it] = np.mean(np.sqrt(xtmp))
                    RMSE[i_a] = np.mean(x)

                d_RMSE[f'F{F}G{G}obs{obs_type}vary{vary}'] = RMSE.copy()
                d_KS[f'F{F}G{G}vary{vary}'] = np.copy(KS)
                d_LE1[f'F{F}G{G}vary{vary}'] = LE1.copy()
                d_n0[f'F{F}G{G}vary{vary}'] = n0
                        
    plot_scatter_KS(d_RMSE, d_LE1, d_KS, d_n0)
    plot_scatter_LE1(d_RMSE, d_LE1, d_KS, d_n0)

analysis()