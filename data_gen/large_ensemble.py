import dapper
import core  
import numpy as np

def set_up_dyn_and_obs(model, n_obs, dyn_noise = 0., obs_noise = 1.):
    # Nx is size of state vector
    Nx = model.M
    Dyn = {
        'M'     : Nx,
        'model' : dapper.with_rk4(model.dxdt,autonom=True),
        'noise' : dyn_noise,
        }
    #
    if n_obs == Nx:
        jj = np.arange(Nx)
    else:
        jj = np.zeros(1, dtype = np.int)
        while len(jj) < n_obs:
            jj = np.unique(np.append(jj, np.random.randint(1, high = Nx)))
    #
    print(obs_noise)
    Obs = dapper.partial_Id_Obs(Nx,jj)
    #
    Obs['noise'] = dapper.GaussRV(M=Nx, C=obs_noise, mu=0.)
    #
    # Obs['localizer'] = localization.partial_direct_obs_nd_loc_setup((Nx//2, ), (2, ), jj, periodic = True)
    return Dyn, Obs, jj

def get_stable_init(Dyn, Obs, K = 500):
    t = dapper.Chronology(0.05, dkObs=1, K=2000, BurnIn=0)
    #
    Nx = Dyn['M']
    x0 = np.random.rand(Nx)
    #
    X0 = dapper.GaussRV(M=Nx, C=0., mu=x0)
    #
    HMM = dapper.HiddenMarkovModel(Dyn,Obs,t,X0)
    #
    xx,yy = dapper.simulate(HMM)
    #
    return xx[-1, :]

def large_ensemble_check(args):
    Nx, F, G, N = args[0], args[1], args[2], args[3]
    print(Nx, F, G, N)
    ntimes = 5

    obs_noise = np.zeros(2*Nx)
    model = core.model(Nx, F, G, alpha = 1., gamma = 1.)
    std = model.std(str(Nx)+str(F)+str(G))
    obs_noise[:Nx] = 0.05*(std[0]**2)
    obs_noise[Nx:] = 0.05*(std[1]**2)
    Dyn, Obs, _ = set_up_dyn_and_obs(model, n_obs = model.M, obs_noise = obs_noise)
    x0 = get_stable_init(Dyn, Obs)
    # get the truth
    t = dapper.Chronology(0.05, dkObs=1, K = 20000, BurnIn=0)
    X0 = dapper.GaussRV(M=model.M, C=0., mu=x0)
    HMM = dapper.HiddenMarkovModel(Dyn,Obs,t,X0)
    #
    xx,yy = dapper.simulate(HMM)
    X0 = dapper.GaussRV(M=model.M, C=obs_noise, mu=x0)
    #
    HMM = dapper.HiddenMarkovModel(Dyn,Obs,t,X0)    
    config = dapper.EnKF_N(N)
    #
    stats = {}
    for it in range(ntimes):
        stat = config.assimilate(HMM,xx,yy)
        stats['rmse_x_f'+str(it)] = np.sqrt(np.mean(stat.err.f[:, :Nx]**2, axis = 1))
        stats['rmse_theta_f'+str(it)] = np.sqrt(np.mean(stat.err.f[:, Nx:]**2, axis = 1))
        stats['rmse_f'+str(it)] = stat.rmse.f

        stats['rmse_x_a'+str(it)] = np.sqrt(np.mean(stat.err.a[:, :Nx]**2, axis = 1))
        stats['rmse_theta_a'+str(it)] = np.sqrt(np.mean(stat.err.a[:, Nx:]**2, axis = 1))
        stats['rmse_a'+str(it)] = stat.rmse.a        
    #
    return Nx, F, G, N, stats

def setup_large_ensemble():
    configs = []
    ensemble_sizes = range(2, 20)
    for Nx in [18]:
        for F, G in zip([10, 10, 0], [0, 10, 10]):
            for N in ensemble_sizes:
                configs += [list([Nx, F, G, N])]

    import multiprocessing
    with multiprocessing.Pool(2) as p:
        for Nx, F, G, N,  stats in p.imap_unordered(large_ensemble_check, configs):
            filename = 'Nx'+str(Nx)+'_F'+str(F)+'_G'+str(G)+'_N'+str(N)+'.npz'
            np.savez(filename, **stats)

if __name__ == '__main__':
    setup_large_ensemble()
