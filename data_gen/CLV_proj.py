import dapper
import numpy as np
import core
import multiprocessing


def simulate(x0, Nx, dt, nt, F, G, alpha = 1., gamma = 1.):
    t = dapper.tools.chronos.Chronology(dt, dkObs=1, K=nt,BurnIn=0)
    VL = core.model(nX=Nx, F = F, G = G, alpha = alpha, gamma = gamma)
    Dyn = {
        'M'     : VL.M,
        'model' : dapper.mods.integration.with_rk4(VL.dxdt,autonom=True),
        'noise' : 0,
    }
    X0 = dapper.tools.randvars.GaussRV(mu=x0, C = 0.)
    jj = np.arange(VL.M)
    Obs = dapper.mods.utils.partial_Id_Obs(VL.M,jj)
    Obs['noise'] = 1.

    HMM = dapper.mods.HiddenMarkovModel(Dyn,Obs,t,X0)

    xx,yy = HMM.simulate()
    return xx, VL

def CLV_experiments(args):
    nx, F, G, alpha, gamma = args[0], args[1], args[2], args[3], args[4]
    spin_up_time = 100
    dt = 0.05
    nt = int(spin_up_time/dt)
    x0, VL = simulate(x0 = np.random.rand(2*nx), Nx = nx, dt = dt, nt = nt, F = F, G=G, alpha = alpha, gamma = gamma)
    x0 = x0[-1, :]
    t = 1000
    nt = int(t/dt)
    x, VL = simulate(x0 = x0, Nx = nx, dt = dt, nt = nt, F = F, G=G, alpha = alpha, gamma = gamma)

    # CLV_filename = "C:/Users/ia923171/OneDrive - University of Reading/Documents/VL20/CLVs/"
    # CLV_filename = "../../../Documents/"
    CLV_filename = "CLVs/"
    CLV_filename += f"LE_F_{F}G_{G}alpha_{np.round(alpha, 2)}.npy"
    # CLV_filename += f'LE_F{F}_G{G}_alpha{alpha}_gamma{gamma}.npy'
    CLVs = np.load(CLV_filename)

    proj = np.zeros_like(CLVs)
    for it in range(nt + 1):
        x_norm = np.linalg.norm(x[it, :])
        for i in range(2*nx):
            cos_theta = np.dot(CLVs[it, :, i], x[it, :])/np.linalg.norm(CLVs[it, :, i])/x_norm
            proj[it, :, i] = x[it, :]*cos_theta/x_norm
            # proj[it, :, i] /= np.linalg.norm(proj[it, :, i])

    # proj = np.mean(proj, axis = 0)
    return F, G, alpha, gamma, proj

def setup_CLV(N_process):
    configs = []
    alphas = [0.4, 2.2]
    gammas = [1.]
    # alphas = [1.]
    # gammas = [0.4, 1.0]
    for F, G in zip([10, 10, 0], [0, 10, 10]):
        for alpha in alphas:
            for gamma in gammas:
                configs +=[list([18, F, G, alpha, gamma])]


    with multiprocessing.Pool(N_process) as p:
        for F, G, alpha, gamma, proj in p.imap_unordered(CLV_experiments, configs):
            filename = 'proj_X_F_'+str(F)+'G_'+str(G)+'alpha_'+str(alpha)+'gamma_'+str(gamma)+'.npy'
            np.save(filename, proj)

if __name__ == '__main__':
    dapper.tools.progressbar.disable_progbar = True
    dapper.tools.progressbar.disable_user_interaction = True
    setup_CLV(N_process = 4)