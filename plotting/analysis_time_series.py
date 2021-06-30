import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

def std(label):
    r"""
        Standard deviation of model variable X and $\theta$

        Parameters
        ----------
        label : str
                A string consists of the size of gridpoints, forcing of X and $\theta$
                e.g. '36100' represents 36 gridpoints, F = 10 and G = 0.
    """
    std = {}
    #
    std['18100']  = np.array([3.723, 2.276])
    std['181010'] = np.array([3.528, 4.496])
    std['18010']  = np.array([2.509, 3.741])
    std['1888']   = np.array([2.866, 3.846])
    return std[label]

def plotting(x, y, labels, colors, Nx):

    fig = plt.figure(1, figsize = (10, 8))
    ax = fig.add_subplot(111)
    for i, label in enumerate(labels):
        ax.plot(x[:2000:10], y[i][:2000:10], color = colors[i], marker = 'None', markerfacecolor = 'None', label = label)

    ax.set_yticks(np.arange(0., 0.6, 0.1))
    ax.set_yticklabels(np.round(np.arange(0., 0.6, 0.1), 2), fontsize = 28)
    ax.set_xticks(np.arange(0, 2001, 200))
    ax.set_xticklabels(np.arange(0, 101, 10), fontsize = 28)
    ax.xaxis.grid(True, which='minor')
    ax.set_xlabel('Time units', fontsize=32)
    ax.set_ylabel('nRMSEa', fontsize=32)
    ax.legend(fontsize=24)
    figname = 'time_series'
    fig.savefig(figname+'.pdf')
    subprocess.run(['pdfcrop', figname+'.pdf'])
    os.remove(figname+'.pdf')
    os.rename(figname+'-crop.pdf',figname+'.pdf')

def analysis():
    # ensemble_size = np.arange(2, 15)
    # ensemble_size = np.append(ensemble_size, np.arange(20, 30, 10))
    times = 5
    colors = ['r', 'b', 'g']
    for Nx in [18]:
        y = []
        labels = []
        for F, G in zip([10, 10, 0], [0, 10, 10]):
            # y_element = np.zeros([len(ensemble_size)])
            stds = std(str(Nx)+str(F)+str(G))
            sigma = np.array([0.05*(stds[0]**2), 0.05*(stds[1]**2)])
            # sigma = np.mean(sigma)
            # for N in [40]:
            N = 40
            filename = '../RMSE/Nx'+str(Nx)+'_F'+str(F)+'_G'+str(G)+'_N'+str(N)+'_alpha1.0_gamma1.0_all.npz'
            print(filename)
            f = np.load(filename)
            x = np.zeros([times, 40000])
            for it in range(times):
                x[it, :] = 0.5*(f['rmse_x_a'+str(it)]**2/sigma[0] + f['rmse_theta_a'+str(it)]**2/sigma[1])
                x[it, :] = np.sqrt(x[it, :])
            y_element = x[0, :]
            print(y_element.shape)
            y.append(y_element)
            labels.append('F = '+str(F)+', G = '+str(G))
        print(y)
        plotting(np.arange(0, 20000), y, labels,colors, Nx)

if __name__ == '__main__':
    analysis()