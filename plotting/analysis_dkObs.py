"""Plot module for scatter plot using various observation interval for nRMSEa agaisnt KS entropy and the first Lyapunov exponent.
"""

import matplotlib.pyplot as plt
import numpy as np
import subprocess
import os
def Get_KS(LE): return LE[LE>0].sum()

def plot_scatter_KS(RMSE_a, LE0_a, KS_a, d_n0_a, RMSE_g, LE0_g, KS_g, d_n0_g):
    fig = plt.figure(1, figsize=(12, 8))
    figlegend = plt.figure(4, figsize = (32, 6))
    plt.clf()
    ax = fig.add_subplot(111)
    colors = ['c', 'g', 'b', 'k', 'y', 'm']
    for i_k, dk in enumerate([1, 3, 5, 10]):
        # dk = 9 if i_k == 5 else i_k
        KS = np.array([])
        RMSE = np.array([])

        KS_d = np.array([])
        RMSE_d = np.array([])

        KS_bound = np.array([])
        upper_bound = np.array([])
        for F, G in zip([10, 10, 0], [0, 10, 10]):
            KS_tmp = np.where(LE0_a[f'F{F}G{G}']*0.05*(dk) < np.log(2.), np.nan, KS_a[f'F{F}G{G}'])
            KS_d = np.append(KS_d,  KS_tmp.flatten())
            RMSE_d = np.append(RMSE_d, np.where(KS_tmp == np.nan, np.nan, RMSE_a[f'F{F}G{G}'][:, :, i_k]).flatten() )
            
            KS_tmp = np.where(LE0_g[f'F{F}G{G}']*0.05*(dk) < np.log(2.), np.nan, KS_g[f'F{F}G{G}'])
            KS_d = np.append(KS_d,  KS_tmp.flatten())
            RMSE_d = np.append(RMSE_d, np.where(KS_tmp == np.nan, np.nan, RMSE_g[f'F{F}G{G}'][:, :, i_k]).flatten() )

            KS_tmp = np.where(LE0_a[f'F{F}G{G}']*0.05*(dk) >= np.log(2.), np.nan, KS_a[f'F{F}G{G}'])
            KS = np.append(KS,  KS_tmp.flatten())
            RMSE = np.append(RMSE, np.where(KS_tmp == np.nan, np.nan, RMSE_a[f'F{F}G{G}'][:, :, i_k]).flatten() )
            
            KS_tmp = np.where(LE0_g[f'F{F}G{G}']*0.05*(dk) >= np.log(2.), np.nan, KS_g[f'F{F}G{G}'])
            KS = np.append(KS,  KS_tmp.flatten())
            RMSE = np.append(RMSE, np.where(KS_tmp == np.nan, np.nan, RMSE_g[f'F{F}G{G}'][:, :, i_k]).flatten() )

            KS_bound = np.append(KS_bound,  KS_a[f'F{F}G{G}'].flatten())
            upper_bound = np.append(upper_bound, (np.sqrt(d_n0_a[f'F{F}G{G}']*(np.exp(2*LE0_a[f'F{F}G{G}']*0.05*(dk)) - 1)/36)).flatten() )

            KS_bound = np.append(KS_bound,  KS_g[f'F{F}G{G}'].flatten())
            upper_bound = np.append(upper_bound, (np.sqrt(d_n0_g[f'F{F}G{G}']*(np.exp(2*LE0_g[f'F{F}G{G}']*0.05*(dk)) - 1)/36)).flatten() )

        ax.scatter(np.log(np.array(RMSE_d)), np.log(np.array(KS_d)), marker='o', color=colors[i_k], label = rf'dkObs{dk} (> doubling time)', s=240)
        ax.scatter(np.log(np.array(RMSE)), np.log(np.array(KS)), facecolors='none', color = colors[i_k], label = rf'dkObs{dk} (<= doubling time)', s=240)
        
        ax.scatter(np.log(np.array(upper_bound)), np.log(np.array(KS_bound)), marker='+', color = colors[i_k], label = f'upper bound (dkObs{dk})', s = 240)

    lgnd = figlegend.legend(*ax.get_legend_handles_labels(), 'center', prop = {'size': 24}, ncol=4, handlelength=2.)

    ax.set_xlabel('ln(nRMSEa)', fontsize=32)
    ax.set_ylabel(r'ln($\sigma_{KS}$)', fontsize=32)
    ax.set_ylim([-0.01, 3.2])
    ax.set_yticks(np.arange(0, 3.5, 0.5))
    ax.set_yticklabels(np.round(np.arange(0, 3.5, 0.5), 2), fontsize=28)
    ax.set_xticks(np.arange(-3.2, 2, 1))
    ax.set_xticklabels(np.round(np.arange(-3.2, 2, 1), 2), fontsize=28)

    figlegend.savefig('legend.pdf')
    figname = 'scatter_dkObs_KE'
    fig.savefig(figname+'.pdf')
    subprocess.run(['pdfcrop', figname+'.pdf'])
    os.remove(figname+'.pdf')
    os.rename(figname+'-crop.pdf',figname+'.pdf')

    subprocess.run(['pdfcrop', 'legend.pdf'])
    os.remove('legend.pdf')
    os.rename('legend-crop.pdf','legend.pdf')

def plot_scatter_LE1(RMSE_a, LE0_a, KS_a, d_n0_a, RMSE_g, LE0_g, KS_g, d_n0_g):
    fig = plt.figure(2, figsize=(12, 8))
    figlegend = plt.figure(4, figsize = (32, 6))
    plt.clf()
    ax = fig.add_subplot(111)
    colors = ['c', 'g', 'b', 'k', 'y', 'm']
    for i_k, dk in enumerate([1, 3, 5, 10]):
        # dk = 9 if i_k == 5 else i_k
        LE1 = np.array([])
        RMSE = np.array([])

        LE1_d = np.array([])
        RMSE_d = np.array([])

        LE1_bound = np.array([])
        upper_bound = np.array([])

        for F, G in zip([10, 10, 0], [0, 10, 10]):
            LE1_tmp = np.where(LE0_a[f'F{F}G{G}']*0.05*(dk) < np.log(2.), np.nan, LE0_a[f'F{F}G{G}'])
            LE1_d = np.append(LE1_d,  LE1_tmp.flatten())
            RMSE_d = np.append(RMSE_d, np.where(LE1_tmp == np.nan, np.nan, RMSE_a[f'F{F}G{G}'][:, :, i_k]).flatten() )
            
            LE1_tmp = np.where(LE0_g[f'F{F}G{G}']*0.05*(dk) < np.log(2.), np.nan, LE0_g[f'F{F}G{G}'])
            LE1_d = np.append(LE1_d,  LE1_tmp.flatten())
            RMSE_d = np.append(RMSE_d, np.where(LE1_tmp == np.nan, np.nan, RMSE_g[f'F{F}G{G}'][:, :, i_k]).flatten() )

            LE1_tmp = np.where(LE0_a[f'F{F}G{G}']*0.05*(dk) >= np.log(2.), np.nan, LE0_a[f'F{F}G{G}'])
            LE1 = np.append(LE1,  LE1_tmp.flatten())
            RMSE = np.append(RMSE, np.where(LE1_tmp == np.nan, np.nan, RMSE_a[f'F{F}G{G}'][:, :, i_k]).flatten() )
            
            LE1_tmp = np.where(LE0_g[f'F{F}G{G}']*0.05*(dk) >= np.log(2.), np.nan, LE0_g[f'F{F}G{G}'])
            LE1 = np.append(LE1,  LE1_tmp.flatten())
            RMSE = np.append(RMSE, np.where(LE1_tmp == np.nan, np.nan, RMSE_g[f'F{F}G{G}'][:, :, i_k]).flatten() )

            LE1_bound = np.append(LE1_bound,  LE0_a[f'F{F}G{G}'].flatten())
            upper_bound = np.append(upper_bound, (np.sqrt(d_n0_a[f'F{F}G{G}']*(np.exp(2*LE0_a[f'F{F}G{G}']*0.05*(dk)) - 1)/36)).flatten() )

            LE1_bound = np.append(LE1_bound,  LE0_g[f'F{F}G{G}'].flatten())
            upper_bound = np.append(upper_bound, (np.sqrt(d_n0_g[f'F{F}G{G}']*(np.exp(2*LE0_g[f'F{F}G{G}']*0.05*(dk)) - 1)/36)).flatten() )

        ax.scatter(np.log(np.array(RMSE_d)), np.log(np.array(LE1_d)), marker='o', color=colors[i_k], label = rf'dkObs{dk} (> doubling time)',s=240)
        ax.scatter(np.log(np.array(RMSE)), np.log(np.array(LE1)), facecolors='none', color = colors[i_k], label = rf'dkObs{dk} (<= doubling time)',s=240)

        ax.scatter(np.log(np.array(upper_bound)), np.log(np.array(LE1_bound)), marker='+', color = colors[i_k], label = f'upper bound (dkObs{dk})', s = 240)

    lgnd = figlegend.legend(*ax.get_legend_handles_labels(), 'center', prop = {'size': 24}, ncol=4, handlelength=2.)

    ax.set_xlabel('ln(nRMSEa)', fontsize=32)
    ax.set_ylabel(r'ln($\lambda_1$)', fontsize=32)
    ax.set_ylim([-1.05, 1.5])
    ax.set_yticks(np.arange(-1, 2., 0.5))
    ax.set_yticklabels(np.round(np.arange(-1, 2., 0.5), 2), fontsize=28)
    ax.set_xticks(np.arange(-3.2, 2, 1))
    ax.set_xticklabels(np.round(np.arange(-3.2, 2, 1), 2), fontsize=28)

    figlegend.savefig('legend.pdf')
    figname = 'scatter_dkObs_LE1'
    fig.savefig(figname+'.pdf')
    subprocess.run(['pdfcrop', figname+'.pdf'])
    os.remove(figname+'.pdf')
    os.rename(figname+'-crop.pdf',figname+'.pdf')

    subprocess.run(['pdfcrop', 'legend.pdf'])
    os.remove('legend.pdf')
    os.rename('legend-crop.pdf','legend.pdf')

def get_RMSE_KS(alphas, gammas, arr_dkObs):
    times = 5
    obs_type = 'all'
    d_RMSE = {}
    d_KS = {}
    d_LE0 = {}
    d_n0 = {}
    for F, G in zip([10, 10, 0], [0, 10, 10]):
        RMSE = np.zeros([len(alphas), len(gammas), len(arr_dkObs)])
        KS = np.zeros([len(alphas), len(gammas)])
        LE0 = np.zeros([len(alphas), len(gammas)])
        n0 = np.zeros([len(alphas), len(gammas)])
        for i_a, alpha in enumerate(alphas):
            for i_g, gamma in enumerate(gammas):
                filename = f'../std/std_F{F}_G{G}_alpha{np.round(alpha, 3)}_gamma{np.round(gamma, 3)}.npy'
                std = np.load(filename)
                filename = f'../LEs/LE_F{F}_G{G}_alpha{np.round(alpha, 3)}_gamma{np.round(gamma, 3)}.npy'
                LE = np.load(filename)

                KS[i_a, i_g] = Get_KS(LE)
                LE0[i_a, i_g] = max(LE)
                n0[i_a, i_g] = np.sum(LE>0)

                if KS[i_a, i_g] < 1:
                    KS[i_a, i_g] = np.nan
                    LE0[i_a, i_g] = np.nan
                    RMSE[i_a, i_g, :] = np.nan
                    continue

                for i_dk, dkObs in enumerate(arr_dkObs):
                    filename = f'Nx18_F{F}_G{G}_N40_alpha{np.round(alpha, 3)}_gamma{np.round(gamma, 3)}_dkObs{dkObs}_{obs_type}.npz'
                    if dkObs == 1:
                        filename = f'../RMSE/Nx18_F{F}_G{G}_N40_alpha{np.round(alpha, 3)}_gamma{np.round(gamma, 3)}_{obs_type}.npz'
                    stats = np.load(filename)

                    if abs(std[1]) <= 1e-6: std[1] = 0.01
                    if abs(std[0]) <= 1e-6: std[0] = 0.01
                    sigma = np.array([0.05*(std[0]**2), 0.05*(std[1]**2)])
                    nt = len(stats['rmse_x_a0'])
                    x = np.zeros([times, nt])
                    for it in range(times):
                        x[it, :] = 0.5*(stats['rmse_x_a'+str(it)]**2/sigma[0] + stats['rmse_theta_a'+str(it)]**2/sigma[1])
                        x[it, :] = np.sqrt(x[it, :])
                    RMSE[i_a, i_g, i_dk] = np.nanmean(x[:, 2000:])

        d_RMSE[f'F{F}G{G}'] = RMSE
        d_KS[f'F{F}G{G}'] = KS
        d_LE0[f'F{F}G{G}'] = LE0
        d_n0[f'F{F}G{G}'] = n0

    return d_RMSE, d_KS, d_LE0, d_n0

RMSE_a, KS_a, LE0_a, d_n0_a = get_RMSE_KS(alphas=np.arange(0.1, 3., 0.1), gammas = [1.0], arr_dkObs = [1, 3, 5, 10])
RMSE_g, KS_g, LE0_g, d_n0_g = get_RMSE_KS(alphas=[1.], gammas = np.arange(0.3, 1.8, 0.05), arr_dkObs = [1, 3, 5, 10])

plot_scatter_KS(RMSE_a, LE0_a, KS_a, d_n0_a, RMSE_g, LE0_g, KS_g, d_n0_g)
plot_scatter_LE1(RMSE_a, LE0_a, KS_a, d_n0_a, RMSE_g, LE0_g, KS_g, d_n0_g)