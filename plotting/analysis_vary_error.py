import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
def Get_KS(LE): return LE[LE>0].sum()

def plot_scatter_KS(RMSE_a, LE0_a, KS_a, d_n0_a, RMSE_g, LE0_g, KS_g, d_n0_g):
    fig = plt.figure(1, figsize=(12, 8))
    figlegend = plt.figure(2, figsize = (28, 6))
    plt.clf()
    ax = fig.add_subplot(111)
    colors = ['c', 'g', 'b', 'k', 'y', 'm']
    for i_k in range(6):
        KS = np.array([])
        RMSE = np.array([])

        for F, G in zip([10, 10, 0], [0, 10, 10]):
            KS = np.append(KS,  KS_a[f'F{F}G{G}'].flatten())
            RMSE = np.append(RMSE, RMSE_a[f'F{F}G{G}'][:, :, i_k].flatten() )
            
            KS = np.append(KS,  KS_g[f'F{F}G{G}'].flatten())
            RMSE = np.append(RMSE, RMSE_g[f'F{F}G{G}'][:, :, i_k].flatten() )

        ax.scatter(np.log(np.array(RMSE)), np.log(np.array(KS)), color = colors[i_k], label = f'{5 + i_k}% Variance', s = 240)

    KS = np.array([])
    bound = np.array([])
    for F, G in zip([10, 10, 0], [0, 10, 10]):
        KS = np.append(KS,  KS_a[f'F{F}G{G}'].flatten())
        bound = np.append(bound, np.sqrt(d_n0_a[f'F{F}G{G}']*(np.exp(2*LE0_a[f'F{F}G{G}']*0.05) - 1)/36) )

        KS = np.append(KS,  KS_g[f'F{F}G{G}'].flatten())
        bound = np.append(bound, np.sqrt(d_n0_g[f'F{F}G{G}']*(np.exp(2*LE0_g[f'F{F}G{G}']*0.05) - 1)/36) )

    ax.scatter(np.log(np.array(bound)), np.log(np.array(KS)), marker='+', color = 'r', label = f'upper bound', s = 240)

    lgnd = figlegend.legend(*ax.get_legend_handles_labels(), 'center', prop = {'size': 20}, ncol=4, handlelength=2., fontsize=12)

    ax.set_xlabel('ln(nRMSEa)', fontsize=32)
    ax.set_ylabel(r'ln($\sigma_{KS}$)', fontsize=32)
    ax.set_ylim([-0.1, 3.2])
    ax.set_yticks(np.arange(0, 3.5, 0.5))
    ax.set_yticklabels(np.round(np.arange(0, 3.5, 0.5), 2), fontsize=28)
    ax.set_xticks(np.arange(-3, -0.8, 0.5))
    ax.set_xticklabels(np.round(np.arange(-3, -0.8, 0.5), 2), fontsize=28)

    figlegend.savefig('legend.pdf')
    figname = f'KS_scatter_vary_error'
    fig.savefig(figname+'.pdf')
    subprocess.run(['pdfcrop', figname+'.pdf'])
    os.remove(figname+'.pdf')
    os.rename(figname+'-crop.pdf',figname+'.pdf')

    subprocess.run(['pdfcrop', 'legend.pdf'])
    os.remove('legend.pdf')
    os.rename('legend-crop.pdf','legend.pdf')

def plot_scatter_LE0(RMSE_a, LE0_a, KS_a, d_n0_a, RMSE_g, LE0_g, KS_g, d_n0_g):
    fig = plt.figure(3, figsize=(12, 8))
    figlegend = plt.figure(4, figsize = (32, 6))
    plt.clf()
    ax = fig.add_subplot(111)
    colors = ['c', 'g', 'b', 'k', 'y', 'm']
    for i_k in range(6):
        LE0 = np.array([])
        RMSE = np.array([])

        for F, G in zip([10, 10, 0], [0, 10, 10]):
            LE0_tmp = np.where(KS_a[f'F{F}G{G}'] < 1, np.nan, LE0_a[f'F{F}G{G}'])
            LE0 = np.append(LE0,  LE0_tmp.flatten())
            RMSE = np.append(RMSE, RMSE_a[f'F{F}G{G}'][:, :, i_k].flatten() )
            
            LE0 = np.append(LE0,  LE0_g[f'F{F}G{G}'].flatten())
            RMSE = np.append(RMSE, RMSE_g[f'F{F}G{G}'][:, :, i_k].flatten() )

        ax.scatter(np.log(np.array(RMSE)), np.log(np.array(LE0)), color = colors[i_k], label = f'{5 + i_k}% Var', s = 240)

    LE0 = np.array([])
    bound = np.array([])
    for F, G in zip([10, 10, 0], [0, 10, 10]):
        LE0 = np.append(LE0,  LE0_a[f'F{F}G{G}'].flatten())
        bound = np.append(bound, np.sqrt(d_n0_a[f'F{F}G{G}']*(np.exp(2*LE0_a[f'F{F}G{G}']*0.05) - 1)/36) )

        LE0 = np.append(LE0,  LE0_g[f'F{F}G{G}'].flatten())
        bound = np.append(bound, np.sqrt(d_n0_g[f'F{F}G{G}']*(np.exp(2*LE0_g[f'F{F}G{G}']*0.05) - 1)/36) )

    ax.scatter(np.log(np.array(bound)), np.log(np.array(LE0)), marker='+', color = 'r', label = f'upper bound', s = 240)

    lgnd = figlegend.legend(*ax.get_legend_handles_labels(), 'center', prop = {'size': 20}, ncol=4, handlelength=2., fontsize=12)

    ax.set_xlabel('ln(nRMSEa)', fontsize=32)
    ax.set_ylabel(r'ln($\lambda_1$)', fontsize=32)
    ax.set_ylim([-1., 1.6])
    ax.set_yticks(np.arange(-1, 2., 0.5))
    ax.set_yticklabels(np.round(np.arange(-1, 2., 0.5), 2), fontsize=28)
    ax.set_xticks(np.arange(-3, -0.8, 0.5))
    ax.set_xticklabels(np.round(np.arange(-3, -0.8, 0.5), 2), fontsize=28)

    figlegend.savefig('legend.pdf')

    figname = f'LE1_scatter_vary_error'
    fig.savefig(figname+'.pdf')
    
    subprocess.run(['pdfcrop', figname+'.pdf'])
    os.remove(figname+'.pdf')
    os.rename(figname+'-crop.pdf',figname+'.pdf')

    subprocess.run(['pdfcrop', 'legend.pdf'])
    os.remove('legend.pdf')
    os.rename('legend-crop.pdf','legend.pdf')

def get_RMSE_KS(alphas, gammas, coeffs):
    times = 5
    obs_type = 'all'
    d_RMSE = {}
    d_KS = {}
    d_LE0 = {}
    d_n0 = {}
    for F, G in zip([10, 10, 0], [0, 10, 10]):
        RMSE = np.zeros([len(alphas), len(gammas), len(coeffs)])
        KS = np.zeros([len(alphas), len(gammas)])
        LE0 = np.zeros([len(alphas), len(gammas)])
        n0 = np.zeros([len(alphas), len(gammas)])
        RMSE[:, :, :] = np.nan
        for i_a, alpha in enumerate(alphas):
            for i_g, gamma in enumerate(gammas):
                filename = f'../std/std_F{F}_G{G}_alpha{np.round(alpha, 3)}_gamma{np.round(gamma, 3)}.npy'
                std = np.load(filename)
                filename = f'../LEs/LE_F{F}_G{G}_alpha{np.round(alpha, 3)}_gamma{np.round(gamma, 3)}.npy'
                LE = np.load(filename)

                KS[i_a, i_g] = Get_KS(LE)
                LE0[i_a, i_g] = LE[0]
                n0[i_a, i_g] = np.sum(LE>0)

                if KS[i_a, i_g] < 1:
                    KS[i_a, i_g] = np.nan
                    LE0[i_a, i_g] =np.nan
                    RMSE[i_a, i_g, :] = np.nan
                    continue

                for i_c, coeff in enumerate(coeffs):
                    filename = f'vary_error/Nx18_F{F}_G{G}_N40_alpha{np.round(alpha, 3)}_gamma{np.round(gamma, 3)}_{obs_type}_{coeff}.npz'
                    if coeff == 0.05:
                        filename = f'../RMSE/Nx18_F{F}_G{G}_N40_alpha{np.round(alpha, 3)}_gamma{np.round(gamma, 3)}_{obs_type}.npz'
                    stats = np.load(filename)

                    if abs(std[1]) <= 1e-6: std[1] = 0.01
                    if abs(std[0]) <= 1e-6: std[0] = 0.01
                    sigma = np.array([coeff*(std[0]**2), coeff*(std[1]**2)])
                    nt = len(stats['rmse_x_a0'])
                    x = np.zeros([times, nt])
                    for it in range(times):
                        x[it, :] = 0.5*(stats['rmse_x_a'+str(it)]**2/sigma[0] + stats['rmse_theta_a'+str(it)]**2/sigma[1])
                        x[it, :] = np.sqrt(x[it, :])
                    RMSE[i_a, i_g, i_c] = np.nanmean(x[:, 20000:])


        d_RMSE[f'F{F}G{G}'] = RMSE
        d_KS[f'F{F}G{G}'] = KS
        d_LE0[f'F{F}G{G}'] = LE0
        d_n0[f'F{F}G{G}'] = n0

    return d_RMSE, d_KS, d_LE0, d_n0

RMSE_a, KS_a, LE0_a, d_n0_a = get_RMSE_KS(alphas=np.arange(0.1, 3., 0.1), gammas = [1.0], coeffs = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
RMSE_g, KS_g, LE0_g, d_n0_g = get_RMSE_KS(alphas=[1.], gammas = np.arange(0.3, 1.8, 0.05), coeffs = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1])

# print(LE0_a.shape)
plot_scatter_KS(RMSE_a, LE0_a, KS_a, d_n0_a, RMSE_g, LE0_g, KS_g, d_n0_g)
plot_scatter_LE0(RMSE_a, LE0_a, KS_a, d_n0_a, RMSE_g, LE0_g, KS_g, d_n0_g)
