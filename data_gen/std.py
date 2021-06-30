import numpy as np
import core
import dapper

def simulate(x0, Nx, dt, nt, F, G, sd0, alpha = 1., gamma = 1.):
    t = dapper.Chronology(dt, dkObs=1, K=nt,BurnIn=0)
    VL = core.model(nX=Nx, F = F, G = G, alpha = alpha, gamma = gamma)
    Dyn = {
        'M'     : VL.M,
        'model' : dapper.with_rk4(VL.dxdt,autonom=True),
        'noise' : 0,
    }
    X0 = dapper.GaussRV(mu=x0, C = 0.)
    jj = np.arange(VL.M)
    Obs = dapper.partial_Id_Obs(VL.M,jj)
    Obs['noise'] = 0.

    HMM = dapper.HiddenMarkovModel(Dyn,Obs,t,X0)
    dapper.seed(sd0)
    xx,yy = dapper.simulate(HMM)
    return xx

def range_check(args):
    F, G, alpha, gamma = args[0], args[1], args[2], args[3]
    Nx = 18
    sd0 = dapper.seed_init(8)
    dapper.seed(sd0)
    xx = simulate(np.random.rand(2*Nx), Nx, 0.05, 2000, F, G, sd0, alpha = alpha, gamma = gamma)
    x0 = xx[-1, :]
    xx = simulate(x0, Nx, 0.05, 500000, F, G, sd0, alpha = alpha, gamma = gamma)

    std = np.zeros(2)
    std[0] = np.std(xx[:, :Nx].flatten())
    std[1] = np.std(xx[:, Nx:].flatten())             
    return F, G, alpha, gamma, std

def setup_std(alphas, gammas):
    configs = []
    for F, G in zip([10, 10, 0], [0, 10, 10]):
        for alpha in alphas:
            for gamma in gammas:
                configs +=[list([F, G, alpha, gamma])]

    import multiprocessing
    with multiprocessing.Pool(10) as p:
        for F, G, alpha, gamma, std in p.imap_unordered(range_check, configs):
            filename = f'std_F{F}_G{G}_alpha{np.round(alpha, 3)}_gamma{np.round(gamma, 3)}.npy'
            np.save(filename, std)

if __name__ == '__main__':
    setup_std(alphas=np.arange(0., 3., 0.1), gammas=np.arange(0.3, 1.8, 0.05))