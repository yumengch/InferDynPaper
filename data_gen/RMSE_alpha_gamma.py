import dapper
import core  
import numpy as np
import dapper.da_methods
def set_up_dyn_and_obs(model, obs_noise, obs_type = 'all', dyn_noise = 0.):
    # Nx is size of state vector
    Nx = model.M
    Dyn = {
        'M'     : Nx,
        'model' : dapper.mods.integration.with_rk4(model.dxdt,autonom=True),
        'noise' : dyn_noise,
        }
    #
    print(obs_type)
    if obs_type == 'X':
        jj = np.arange(model.nX)
        Obs = dapper.mods.utils.partial_Id_Obs(Nx,jj)
        Obs['noise'] = dapper.tools.randvars.GaussRV(M=model.nX, C=obs_noise[:model.nX], mu=0.)
    elif obs_type == 'theta':
        jj = np.arange(model.nX, Nx)
        Obs = dapper.mods.utils.partial_Id_Obs(Nx,jj)
        Obs['noise'] = dapper.tools.randvars.GaussRV(M=model.nX, C=obs_noise[model.nX:], mu=0.)
    else:
        jj = np.arange(Nx)
        Obs = dapper.mods.utils.partial_Id_Obs(Nx,jj)
        Obs['noise'] = dapper.tools.randvars.GaussRV(M=Nx, C=obs_noise, mu=0.)

    return Dyn, Obs, jj

def get_stable_init(Dyn, Obs, sd0, K = 2000):
    t = dapper.mods.Chronology(0.05, dkObs=1, K = K,BurnIn=0)
    #
    Nx = Dyn['M']
    dapper.set_seed(sd0)
    x0 = np.random.rand(Nx)
    #
    X0 = dapper.tools.randvars.GaussRV(M=Nx, C=0., mu=x0)
    #
    HMM = dapper.mods.HiddenMarkovModel(Dyn,Obs,t,X0)
    #
    dapper.set_seed(sd0)
    xx,yy = HMM.simulate()
    #
    return xx[-1, :]

def single_obs(args):
    F, G, alpha, gamma = args[0], args[1], args[2], args[3]
    
    N = 40
    Nx = 18
    ntimes = 3
    obs_type = 'all'
    obs_noise = np.zeros(2*Nx)
    model = core.model(Nx, F, G, alpha = alpha, gamma = gamma)

    std = np.load(f'stds/std_F{F}_G{G}_alpha{np.round(alpha, 3)}_gamma{np.round(gamma, 3)}.npy')
    if abs(std[1]) <= 1e-6: std[1] = 0.01
    if abs(std[0]) <= 1e-6: std[0] = 0.01
    obs_noise[:Nx] = 0.05*(std[0]**2)
    obs_noise[Nx:] = 0.05*(std[1]**2)
    filename = f'LEs/LE_F{F}_G{G}_alpha{np.round(alpha, 3)}_gamma{np.round(gamma, 3)}.npy'
    LE = np.load(filename)
    print(F, G, alpha, gamma, 'LE:', LE[0], 'obs:', obs_noise[0], obs_noise[-1])

    sd0 = dapper.set_seed(8)
    Dyn, Obs, _ = set_up_dyn_and_obs(model, obs_noise = obs_noise, obs_type = obs_type)
    x0 = get_stable_init(Dyn, Obs, sd0)
    # get the truth
    t = dapper.mods.Chronology(0.05, dkObs=1, K = 20000, BurnIn=0)
    X0 = dapper.tools.randvars.GaussRV(M=model.M, C=0., mu=x0)
    HMM = dapper.mods.HiddenMarkovModel(Dyn,Obs,t,X0)
    #
    dapper.set_seed(sd0 + 10)
    xx,yy = HMM.simulate()
    X0 = dapper.tools.randvars.GaussRV(M=model.M, C=obs_noise, mu=x0)
    #
    HMM = dapper.mods.HiddenMarkovModel(Dyn,Obs,t,X0)    
    config = dapper.da_methods.EnKF_N(N)
    #
    stats = {}
    for it in range(ntimes):
        dapper.set_seed(sd0 + it + 13)
        config.assimilate(HMM,xx,yy)
        stats['rmse_x_f'+str(it)] = np.sqrt(np.mean(config.stats.err.f[:, :Nx]**2, axis = 1))
        stats['rmse_theta_f'+str(it)] = np.sqrt(np.mean(config.stats.err.f[:, Nx:]**2, axis = 1))
        stats['rmse_f'+str(it)] = config.stats.err.rms.f

        stats['rmse_x_a'+str(it)] = np.sqrt(np.mean(config.stats.err.a[:, :Nx]**2, axis = 1))
        stats['rmse_theta_a'+str(it)] = np.sqrt(np.mean(config.stats.err.a[:, Nx:]**2, axis = 1))
        stats['rmse_a'+str(it)] = config.stats.err.rms.a        

    return F, G, alpha, gamma, obs_type, stats

def setup_single_obs(alphas, gammas):
    configs = []
    for F, G in zip([10, 10, 0], [0, 10, 10]):
        for alpha in alphas:
           for gamma in gammas:
                filename = f'LEs/LE_F{F}_G{G}_alpha{np.round(alpha, 3)}_gamma{np.round(gamma, 3)}.npy'
                LE = np.load(filename)
                if LE[0] >= 1e-6:
                    configs += [list([F, G, alpha, gamma])]

    import multiprocessing
    with multiprocessing.Pool(3) as p:
        for F, G, alpha, gamma, obs_type, stats in p.imap_unordered(single_obs, configs):
            filename = 'Nx18_F'+str(F)+'_G'+str(G)+'_N40_alpha'+str(np.round(alpha,2))+'_gamma'+str(np.round(gamma,3))+'_'+obs_type+'.npz'
            np.savez(filename, **stats)

if __name__ == '__main__':
    dapper.tools.progressbar.disable_progbar = True
    dapper.tools.progressbar.disable_user_interaction = True
    setup_single_obs(alphas=np.round(np.arange(0.1, 3., 0.1), 3), gammas=np.round(np.arange(0.3, 1.8, 0.05), 3))