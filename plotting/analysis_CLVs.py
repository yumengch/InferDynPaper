"""amplitude of CLVs for line plot."""
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import matplotlib.cm as cm
from matplotlib import colors
import matplotlib.markers as mrks

def plot(field, F, G, alpha, gamma, is_alpha):
    # cmap2 = cm.get_cmap(,lut=15)
    # cmap2.set_under("k")
    vmin = 0.04; vmax = 0.34
    # vmin = -0.003; vmax = 0.003
    fig = plt.figure(1, figsize = (10, 8))
    
    fig.clf()
    ax = fig.add_subplot(111)
    # norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.25, vmax=vmax)
    im = ax.pcolormesh(field.T, cmap = 'YlGnBu', vmin = vmin, vmax = vmax)
    ax.set_xlabel('State index', fontsize=32)
    ax.set_ylabel('CLV vector index', fontsize=32)

    ax.set_xticks(np.arange(0.5, 36.5, 4))
    ax.set_xticklabels(np.round(np.arange(1, 37, 4), 2), fontsize=28)
    ax.set_yticks(np.arange(0.5, 36.5, 4))
    ax.set_yticklabels(np.round(np.arange(1, 37, 4), 2), fontsize=28)
   
    LE_filename=f"../LEs/LE_F{F}_G{G}_alpha{alpha}_gamma{gamma}.npy"
    LEs = np.load(LE_filename)
    n0 = np.sum(LEs>0)
    n0 += (abs(LEs[n0]) < abs(LEs[n0-1]))
    print(n0)
    ax.axhline(n0-0.5, color ='r')

    if is_alpha:
        figname = f'CLV_proj_F{F}_G{G}_alpha{alpha}'
    else:
        figname = f'CLV_proj_F{F}_G{G}_gamma{gamma}'
    fig.savefig(figname+'.pdf')

    figcolorbar = plt.figure(2)
    ax = figcolorbar.add_subplot(111)
    figcolorbar.colorbar(im, ax = ax, orientation = 'horizontal')
    ax.remove()
    figcolorbar.savefig('colorbar.pdf')

    subprocess.run(['pdfcrop', figname+'.pdf'])
    os.remove(figname+'.pdf')
    os.rename(figname+'-crop.pdf',figname+'.pdf')

    subprocess.run(['pdfcrop', 'colorbar.pdf'])
    os.remove( 'colorbar.pdf')
    os.rename('colorbar-crop.pdf','colorbar.pdf')

def lineplot(d_field, alpha, gamma, is_alpha):
    fig = plt.figure(1, figsize = (12, 8))
    figlegend = plt.figure(4, figsize = (32, 6))
    fig.clf()
    ax = fig.add_subplot(111)

    markers = {}
    markers[f'F10G0'] = mrks.MarkerStyle('o', 'none')
    markers[f'F10G10'] = mrks.MarkerStyle('^', 'none')
    markers[f'F0G10'] = mrks.MarkerStyle('x', 'none')
    colors = {f'F10G0':'r', f'F10G10':'b',f'F0G10':'g'}

    for F, G in zip([10, 10, 0], [10, 0, 10]):
        ax.plot(d_field[f"F{F}G{G}"][:18, :].mean(axis = 0), marker='o', markersize=10, linestyle='-', linewidth = 4., color=colors[f'F{F}G{G}'], label=r"$\mathbf{X}$"+f" (F = {F}, G = {G})")
        ax.plot(d_field[f"F{F}G{G}"][18:, :].mean(axis = 0), marker='x', markersize=10, linestyle='--', linewidth = 4., color=colors[f'F{F}G{G}'], label=r"$\mathbf{\theta}$"+f" (F = {F}, G = {G})")

        LE_filename=f"../LEs/LE_F{F}_G{G}_alpha{alpha}_gamma{gamma}.npy"
        LEs = np.load(LE_filename)
        n0 = np.sum(LEs>0)
        if n0 > 0:
            n0 += (abs(LEs[n0]) < abs(LEs[n0-1]))
        print(n0, LEs)
        ax.axvline(n0, linestyle=':', linewidth = 4., color =colors[f'F{F}G{G}'], label=r"$n_0$"+f" (F = {F}, G = {G})")

    ax.set_xlabel('CLV vector index', fontsize=32)
    ax.set_ylabel('Amplitude of the CLVs', fontsize=32)

    ax.set_ylim([0.03, 0.26])
    ax.set_yticks(np.arange(0.05, 0.3, 0.05))
    ax.set_yticklabels(np.round(np.arange(0.05, 0.3, 0.05), 2), fontsize=28)
    ax.set_xticks(np.arange(0, 36, 4))
    ax.set_xticklabels(np.round(np.arange(1, 37, 4), 2), fontsize=28)

    lgnd = figlegend.legend(*ax.get_legend_handles_labels(), 'center', prop = {'size': 24}, ncol=3, handlelength=2.)


    figlegend.savefig('legend.pdf')

    if is_alpha:
        figname = f'CLV_alpha{alpha}'
    else:
        figname = f'CLV_gamma{gamma}'

    fig.savefig(figname+'.pdf')
    subprocess.run(['pdfcrop', figname+'.pdf'])
    os.remove(figname+'.pdf')
    os.rename(figname+'-crop.pdf',figname+'.pdf')

    subprocess.run(['pdfcrop', 'legend.pdf'])
    os.remove('legend.pdf')
    os.rename('legend-crop.pdf','legend.pdf')

def proj(is_alpha):
    alpha = 1.
    gamma = 1.
    
    
    if is_alpha:
        for alpha in [0.4, 2.2]:
            d_field = {}
            for F, G in zip([10, 10, 0], [0, 10, 10]):
                CLV_filename = f'../CLVs/LE_F_{F}G_{G}alpha_{alpha}.npy'
                CLVs = np.load(CLV_filename)
                field = np.mean(abs(CLVs[10000:]), axis = 0)
                for i in range(36):
                    field /= np.linalg.norm(field[:, i])
                # print(field[0, :])
                # plot(field, F, G, alpha, gamma, is_alpha)
                d_field[f"F{F}G{G}"] = field.copy()
            lineplot(d_field, alpha, gamma, is_alpha)
    else:
        for gamma in [0.4, 1.0]:
            d_field = {}
            for F, G in zip([10, 10, 0], [0, 10, 10]):
                CLV_filename = f'../CLVs/LE_F{F}_G{G}_alpha{alpha}_gamma{gamma}.npy'
                CLVs = np.load(CLV_filename)
                field = np.mean(abs(CLVs[10000:]), axis = 0)
                for i in range(36):
                    field /= np.linalg.norm(field[:, i])
                    # field /= np.sum(field[:, i])
                # plot(field, F, G, alpha, gamma, is_alpha)
                d_field[f"F{F}G{G}"] = field.copy()
            lineplot(d_field, alpha, gamma, is_alpha)
proj(True)
proj(False)