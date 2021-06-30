import dapper
import numpy as np
import core
import multiprocessing

def computeCLV(f, fjac, x0, t, t_A_converge):
    """
    Computes the global Lyapunov exponents for a set of ODEs.
    f - ODE function. Must take arguments like f(t, x, p) where x and t are 
        the state and time *now*, and p is a tuple of parameters. If there are 
        no model paramters, p should be set to the empty tuple.
    x0 - Initial position for calculation. Integration of transients will begin 
         from this point.
    t - Array of times over which to calculate LE.
    fjac - Jacobian of f.
    method - (optional) Integration method to be used by scipy.integrate.ode.
    """
    D = len(x0)
    print(D)
    N = len(t)
    dt = t[1] - t[0]

    def dPhi_dt(t, Phi, x):
        """ The variational equation """
        D = len(x)
        rPhi = np.reshape(Phi, (D, D))
        rdPhi = np.dot(fjac(t, x), rPhi)
        return rdPhi.flatten()

    def dSdt(t, S):
        """
        Differential equations for combined state/variational matrix
        propagation. This combined state is called S.
        """
        x = S[:D]
        Phi = S[D:]
        return np.append(f(t,x), dPhi_dt(t, Phi, x))


    # start LE calculation
    LE = np.zeros((N-1, D), dtype=np.float64)
    
    Q = np.zeros([N, D, D])
    R = np.zeros([N-1, D, D])
    CLV = np.zeros([N, D, D])
   
    print("Integrating system for CLV calculation...")
    Q[0, :, :] = 0.1*np.random.rand(D, D)
    Ssol = np.append(x0, Q[0, :, :].flatten())
    for i,(t1,t2) in enumerate(zip(t[:-1], t[1:])):
        dt = t2 - t1
        Ssol_temp = dapper.mods.integration.rk4(dSdt, Ssol, t1, dt)
        # perform QR decomposition on Phi
        rPhi = np.reshape(Ssol_temp[D:], (D, D))
        Q[i+1, :, :],R[i, :, :] = np.linalg.qr(rPhi)
        Ssol = np.append(Ssol_temp[:D], Q[i+1, :, :].flatten())

    print("Extended forward steps for convergence...")
    # continue forward to get some time for A convergence.
    N_A_converge = len(t_A_converge)
    R_converge = np.zeros([N_A_converge - 1, D, D])
    for i,(t1,t2) in enumerate(zip(t_A_converge[:-1], t_A_converge[1:])):
        dt = t2 - t1
        Ssol_temp = dapper.mods.integration.rk4(dSdt, Ssol, t1, dt)
        # perform QR decomposition on Phi
        rPhi = np.reshape(Ssol_temp[D:], (D, D))
        Q_converge, R_converge[i, :, :] = np.linalg.qr(rPhi)
        Ssol = np.append(Ssol_temp[:D], Q_converge.flatten())

    print("Extended back steps for convergence...")
    # backward to converge A
    A = np.triu(np.random.rand(D,D))
    A = A/np.linalg.norm(A,axis=0,keepdims=True)
    for i,(t1,t2) in enumerate(zip(t_A_converge[-2::-1], t_A_converge[-1:0:-1])):
        A = np.linalg.solve(R_converge[-1 - i, :, :], A)
        A = A/np.linalg.norm(A,axis=0,keepdims=True)

    # compute CLVs
    print("Compute CLVs...")
    CLV[-1, :, :] = Q[-1, :, :]@A
    for i,(t1,t2) in enumerate(zip(t[-2::-1], t[-1:0:-1])):
        # dt = t2 - t1
        A = np.linalg.solve(R[-1 - i, :, :], A)
        # perform QR decomposition on Phi
        CLV[-1 - i - 1, :, :] = Q[-1 - i - 1, :, :]@A
        A = A/np.linalg.norm(A,axis=0,keepdims=True)
        CLV[-1 - i - 1, :, :] = CLV[-1 - i - 1, :, :]/np.linalg.norm(CLV[-1 - i - 1,:,:],axis=0,keepdims=True)
    return CLV

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
    Obs = dapper.mods.utils.partial_Id_Obs(VL.M, jj)
    Obs['noise'] = 1.

    HMM = dapper.mods.HiddenMarkovModel(Dyn,Obs,t,X0)

    xx,yy = HMM.simulate()
    return xx, VL

def CLV_experiments(args):
    nx, F, G, alpha, gamma = args[0], args[1], args[2], args[3], args[4]
    spin_up_time = 100
    dt = 0.05
    nt = int(spin_up_time/dt)
    x0, VL = simulate(x0 = np.random.rand(2*nx), Nx = nx, dt = dt, nt = nt, F = F, G = G, alpha = alpha, gamma = gamma)
    x0 = x0[-1, :]
    f = lambda t,x : VL.dxdt(x)
    fjac = lambda t,x: VL.d2x_dtdx(x)
    t = np.arange(0., 1000.05, 0.05)
    t_A_converge = np.arange(0., 1000.05, 0.05)
    CLV = computeCLV(f, fjac, x0, t, t_A_converge)

    return F, G, alpha, gamma, CLV

def setup_CLV(N_process):
    configs = []
    alphas = [1.]# np.arange(0, 3, 0.1)
    gammas = [0.4, 1.0]
    for F, G in zip([10, 10, 0], [0, 10, 10]):
        for alpha in alphas:
            for gamma in gammas:
                configs +=[list([18, F, G, alpha, gamma])]


    with multiprocessing.Pool(N_process) as p:
        for F, G, alpha, gamma, CLV in p.imap_unordered(CLV_experiments, configs):
            filename = f'LE_F{F}_G{G}_alpha{alpha}_gamma{gamma}.npy'
            np.save(filename, CLV)

if __name__ == '__main__':
    dapper.tools.progressbar.disable_progbar = True
    dapper.tools.progressbar.disable_user_interaction = True
    setup_CLV(N_process = 4)