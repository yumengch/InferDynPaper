"""This module plots the data using observations for single observations X/theta.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import subprocess
import os

def plot_KS_and_RMSE_gamma(x, y, KS, LE1, Fs, Gs, colors, obs_type):
	"""Plotting the figure for varying gamma.
	"""
    j = {'all': 0, 'X': 1, 'theta': 2}
    for i, (F, G) in enumerate(zip(Fs, Gs)):
        fig = plt.figure(1, figsize = (8, 8))

        figlegend = plt.figure(2, figsize = (20, 4))
        fig.clf()

        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        lns = []
        print(len(y[i][0]), len(KS[i][0]))
        K = KS[i][0, :]
        LE = LE1[i][0, :]
        # 
        K = np.where(K < 1e-6, np.nan, K)
        y[i] = np.where(K == np.nan, np.nan, y[i])
        LE = np.where(K == np.nan, np.nan, LE)
        ln1 = ax1.plot(x, y[i][0], color='b', marker = 'o', linewidth = 4., markersize = 10, markerfacecolor = 'None', linestyle = 'dotted', label = 'X')
        ln2 = ax1.plot(x, y[i][1], color='r', marker = 's', linewidth = 4., markersize = 10, markerfacecolor = 'None', linestyle = 'dotted', label = r'$\theta$')
        ln3 = ax1.plot(x, y[i][2], color='g', marker = 'p', linewidth = 4., markersize = 10, markerfacecolor = 'None', linestyle = 'dotted', label = 'total')
        ln4 = ax2.plot(x, K,   color='k', marker = 'x', linewidth = 4., markersize = 10, markerfacecolor = 'None', linestyle = 'solid',  label = r'$\sigma_{KS}$')
        ln5 = ax2.plot(x, 3*LE,   color='gray', marker = 'x', linewidth = 4., markersize = 10, markerfacecolor = 'None', linestyle = 'solid',  label = r'$3\lambda_1$')

        lns += ln1 + ln2 + ln3 + ln4 + ln5


        ax2.set_ylim([0, 24])
        ax2.set_yticks(np.arange(0, 25, 4))
        ax2.set_yticklabels(np.round(np.arange(0, 25, 4)), fontsize=28)
        if (F, G) == (10, 0):
            ax1.set_ylim([0., 0.9])
            ax1.set_yticks(np.arange(0., 1.0, 0.3))
            ax1.set_yticklabels(np.round(np.arange(0., 1.0, 0.3), 2), fontsize=28)

        if obs_type == 'all' and (F, G) == (10, 0):
            ax1.set_ylim([0.0, 0.3])
            ax1.set_yticks(np.arange(0.0, 0.31, 0.1))
            ax1.set_yticklabels(np.round(np.arange(0., 0.31, 0.1), 2), fontsize=28)

        if (F, G) == (10, 10):
            ax1.set_ylim([0.0, 0.8])
            ax1.set_yticks(np.arange(0.0, 0.9, 0.2))
            ax1.set_yticklabels(np.round(np.arange(0., 0.9, 0.2), 2), fontsize=28)

        if obs_type == 'all' and (F, G) == (10, 10): 
            ax1.set_ylim([0.0, 0.3])
            ax1.set_yticks(np.arange(0.0, 0.31, 0.1))
            ax1.set_yticklabels(np.round(np.arange(0., 0.31, 0.1), 2), fontsize=28)

        if (F, G) == (0, 10): 
            ax1.set_ylim([0., 0.6])
            ax1.set_yticks(np.arange(0., 0.7, 0.15))
            ax1.set_yticklabels(np.round(np.arange(0., 0.7, 0.15), 2), fontsize=28)

        if obs_type == 'all' and (F, G) == (0, 10):   
            ax1.set_ylim([0.0, 0.3])
            ax1.set_yticks(np.arange(0.0, 0.31, 0.1))
            ax1.set_yticklabels(np.round(np.arange(0., 0.31, 0.1), 2), fontsize=28)

        if obs_type == 'all': ax2.set_ylabel(r'$\sigma_{KS}$ or $3\lambda_1$', fontsize=32)
        if obs_type == 'X': ax1.set_ylabel('nRMSEa', fontsize=32)

        if (F, G) == (0, 10): ax1.set_xlabel(r'$\gamma$', fontsize=32)
        ax1.set_xticks(np.arange(0.2, 1.8, 0.2))
        ax1.set_xticklabels(np.round(np.arange(0.2, 1.8, 0.2), 2), rotation=45, fontsize=28)
        ax2.set_xticks(np.arange(0.2, 1.8, 0.2))
        ax2.set_xticklabels(np.round(np.arange(0.2, 1.8, 0.2), 2), rotation=45, fontsize=28)

        filename = f'KS_and_RMSE_F_{F}_G_{G}_gamma_{obs_type}'
        fig.savefig(filename+'.pdf', bbox_inches = 'tight', pad_inches = 0)
        subprocess.run(['pdfcrop', filename+'.pdf'])
        os.remove(filename+'.pdf')
        os.rename(filename+'-crop.pdf',filename+'.pdf')


def plot_KS_and_RMSE_alpha(x, y, KS, LE1, Fs, Gs, colors, obs_type):
	"""Plotting with varying alphas."""
    for i, (F, G) in enumerate(zip(Fs, Gs)):
        fig = plt.figure(1, figsize = (8, 8))

        figlegend = plt.figure(2, figsize = (20, 4))
        fig.clf()
        ax1 = fig.add_subplot(111)

        ax2 = ax1.twinx()
        lns = []

        K = KS[i][:, 0]
        LE = LE1[i][:, 0]
        x_plot = x
        y_plot = y[i]
        K = np.where(K < 1e-6, np.nan, K)
        y_plot = np.where(K == np.nan, np.nan, y_plot)
        LE = np.where(K == np.nan, np.nan, LE)
        ln1 = ax1.plot(x_plot, y_plot[0], color='b', marker = 'o', linewidth = 4., markersize = 10, markerfacecolor = 'None', linestyle = 'dotted', label = 'X')
        ln2 = ax1.plot(x_plot, y_plot[1], color='r', marker = 's', linewidth = 4., markersize = 10, markerfacecolor = 'None', linestyle = 'dotted', label = r'$\theta$')
        ln3 = ax1.plot(x_plot, y_plot[2], color='g', marker = 'p', linewidth = 4., markersize = 10, markerfacecolor = 'None', linestyle = 'dotted', label = 'total')
        ln4 = ax2.plot(x_plot, K,   color='k', marker = 'x', linewidth = 4., markersize = 10, markerfacecolor = 'None', linestyle = 'solid',  label = r'$\sigma_{KS}$')
        ln5 = ax2.plot(x_plot, 3*LE,   color='gray', marker = 'x', linewidth = 4., markersize = 10, markerfacecolor = 'None', linestyle = 'solid',  label = r'$3\lambda_1$')

        lns += ln1 + ln2 + ln3 + ln4 + ln5

        ax2.set_ylim([0, 10])
        ax2.set_yticks(np.arange(0, 11, 2))
        ax2.set_yticklabels(np.round(np.arange(0, 11, 2)), fontsize=28)

        if (F, G) == (10, 0):
            ax1.set_ylim([0.13, 0.30])
            ax1.set_yticks(np.arange(0.13, 0.31, 0.03))
            ax1.set_yticklabels(np.round(np.arange(0.13, 0.31, 0.03), 2), fontsize=28)
        if obs_type == 'all' and (F, G) == (10, 0):
            ax1.set_ylim([0.06, 0.23])
            ax1.set_yticks(np.arange(0.06, 0.24, 0.03))
            ax1.set_yticklabels(np.round(np.arange(0.06, 0.24, 0.03), 2), fontsize=28)


        if (F, G) == (10, 10): 
            ax1.set_ylim([0.17, 0.31])
            ax1.set_yticks(np.arange(0.17, 0.32, 0.03))
            ax1.set_yticklabels(np.round(np.arange(0.17, 0.32, 0.03), 2), fontsize=28)
        if obs_type == 'all' and (F, G) == (10, 10): 
            ax1.set_ylim([0.09, 0.23])
            ax1.set_yticks(np.arange(0.09, 0.24, 0.03))
            ax1.set_yticklabels(np.round(np.arange(0.09, 0.24, 0.03), 2), fontsize=28)


        if (F, G) == (0, 10):  
            ax1.set_ylim([0., 0.3])
            ax1.set_yticks(np.arange(0., 0.31, 0.06))
            ax1.set_yticklabels(np.round(np.arange(0., 0.31, 0.06), 2), fontsize=28)

        if obs_type == 'all' and (F, G) == (0, 10): 
            ax1.set_ylim([0.0, 0.16])
            ax1.set_yticks(np.arange(0.0, 0.17, 0.04))
            ax1.set_yticklabels(np.round(np.arange(0., 0.17, 0.04), 2), fontsize=28)

        if obs_type == 'all': ax2.set_ylabel(r'$\sigma_{KS}$ or $3\lambda_1$', fontsize=32)
        if obs_type == 'X': ax1.set_ylabel('nRMSEa', fontsize=32)

        if (F, G) == (0, 10): ax1.set_xlabel(r'$\alpha$', fontsize=32)
        ax1.set_xticks(np.arange(0.1, 3., 0.4))
        ax1.set_xticklabels(np.round(np.arange(0.1, 3., 0.4), 2), rotation=45, fontsize=28)
        ax2.set_xticks(np.arange(0.1, 3., 0.4))
        ax2.set_xticklabels(np.round(np.arange(0.1, 3., 0.4), 2), rotation=45, fontsize=28)
        labs = [l.get_label() for l in lns]
        figlegend.legend(lns, labs, prop=dict(size=28), ncol = 5)
        figlegend.savefig('legend.pdf')

        filename = f'KS_and_RMSE_F_{F}_G_{G}_alpha_{obs_type}'
        fig.savefig(filename+'.pdf', bbox_inches = 'tight', pad_inches = 0)
        subprocess.run(['pdfcrop', filename+'.pdf'])
        subprocess.run(['pdfcrop', 'legend.pdf'])
        os.remove(filename+'.pdf')
        os.remove( 'legend.pdf')
        os.rename(filename+'-crop.pdf',filename+'.pdf')
        os.rename('legend-crop.pdf','legend.pdf')

def analysis_RMSE(alphas, gammas, Fs, Gs, plot_gamma):
	"""Read generated data from the DA runs for observations for single variables"""
    times = 5
    Nx = 18
    N = 40
    colors = ['r', 'b', 'g']
    plot_gamma = len(gammas) > 1
    plot_type = 'gamma' if plot_gamma else 'alpha'
    KS = []
    LE1 = []

    for obs_type in ['all', 'X', 'theta']:
        y = []
        labels = []
        for F, G in zip(Fs, Gs):
            y_element = np.zeros([3, max(len(gammas), len(alphas))])

            for i_a, alpha in enumerate(alphas):
                for i_g, gamma in enumerate(gammas):
                    j = i_g if plot_gamma else i_a
                    filename = f'../std/std_F{F}_G{G}_alpha{np.round(alpha, 3)}_gamma{np.round(gamma, 3)}.npy'
                    std = np.load(filename)                    
                    if abs(std[1]) <= 1e-6: std[1] = 0.01
                    if abs(std[0]) <= 1e-6: std[0] = 0.01
                    sigma = np.array([0.05*(std[0]**2), 0.05*(std[1]**2)])
                    a = np.around(alpha, 3)
                    g = np.around(gamma, 3)
                    filename = f'../RMSE/Nx{Nx}_F{F}_G{G}_N{N}_alpha{a}_gamma{g}_{obs_type}.npz'
                    f = np.load(filename)
                    nt =len(f['rmse_x_a0'])
                    x = np.zeros([times, nt])
                    for it in range(times):
                        x[it, :] = f['rmse_x_a'+str(it)]/np.sqrt(sigma[0])

                    y_element[0, j] = np.nanmean(x[:, 2000:])
                    for it in range(times):
                        x[it, :] = f['rmse_theta_a'+str(it)]/np.sqrt(sigma[1])
                    y_element[1, j] = np.nanmean(x[:, 2000:])
                    for it in range(times):
                        x[it, :] = 0.5*(f['rmse_x_a'+str(it)]**2/sigma[0] + f['rmse_theta_a'+str(it)]**2/sigma[1])
                        x[it, :] = np.sqrt(x[it, :])
                    y_element[2, j] = np.nanmean(x[:, 2000:])

            y.append(y_element)
            if obs_type == 'all': 
                KS.append(Get_KS(alphas, gammas, F, G))
                LE1.append(Get_LE1(alphas, gammas, F, G))

        if plot_gamma: 
            plot_KS_and_RMSE_gamma(gammas, y, KS, LE1, Fs, Gs, colors, obs_type)
        else:
            plot_KS_and_RMSE_alpha(alphas, y, KS, LE1, Fs, Gs, colors, obs_type)

def Get_KS(alphas, gammas, F, G):
	"""Compute the KS entropy."""
    KS = np.zeros([len(alphas), len(gammas)])
    for i, alpha in enumerate(alphas):
        for j, gamma in enumerate(gammas):
            a = np.around(alpha, 3)
            g = np.around(gamma, 3)
            filename = f'../LEs/LE_F{F}_G{G}_alpha{np.round(alpha, 3)}_gamma{np.round(gamma, 3)}.npy'
            LEs = np.load(filename)
            dt = 0.05
            LE = np.mean(np.log(LEs), axis = 0) /dt if LEs.ndim != 1 else LEs
            
            KS[i, j] = LE[LE>0].sum()
    return KS

def Get_LE1(alphas, gammas, F, G):
	"""Get the first Lyapunov exponent."""
    LE1 = np.zeros([len(alphas), len(gammas)])
    for i, alpha in enumerate(alphas):
        for j, gamma in enumerate(gammas):
            a = np.around(alpha, 3)
            g = np.around(gamma, 3)
            filename = f'../LEs/LE_F{F}_G{G}_alpha{np.round(alpha, 3)}_gamma{np.round(gamma, 3)}.npy'
            LEs = np.load(filename)
            dt = 0.05
            LE = np.mean(np.log(LEs), axis = 0) /dt if LEs.ndim != 1 else LEs
            
            LE1[i, j] = LE[0]

    return LE1

if __name__ == '__main__':
    analysis_RMSE(alphas = [1.0], gammas = np.arange(0.3, 1.45, 0.05), Fs = [10, 10, 0], Gs = [0, 10, 10], plot_gamma = True)
    analysis_RMSE(alphas = np.arange(0.1, 3, 0.1), gammas = [1.0], Fs = [10, 10, 0], Gs = [0, 10, 10], plot_gamma = False)