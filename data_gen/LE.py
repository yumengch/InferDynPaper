import dapper
import numpy as np
import dapper.mods.Lorenz95.core
import core

def computeLE(f, fjac, x0, t, p=(), ttrans=None):
    """
    Computes the global Lyapunov exponents for a set of ODEs.
    f - ODE function. Must take arguments like f(t, x, p) where x and t are 
        the state and time *now*, and p is a tuple of parameters. If there are 
        no model paramters, p should be set to the empty tuple.
    x0 - Initial position for calculation. Integration of transients will begin 
         from this point.
    t - Array of times over which to calculate LE.
    p - (optional) Tuple of model parameters for f.
    fjac - Jacobian of f.
    ttrans - (optional) Times over which to integrate transient behavior.
             If not specified, assumes trajectory is on the attractor.
    method - (optional) Integration method to be used by scipy.integrate.ode.
    """
    D = len(x0)
    print(D)
    N = len(t)
    if ttrans is not None:
        Ntrans = len(ttrans)
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

    # integrate transient behavior
    Phi0 = np.eye(D, dtype=np.float64).flatten()
    #S0 = np.append(x0, Phi0)

    if ttrans is not None:
        print("Integrating transient behavior...")
        #Strans = np.zeros((Ntrans, D*(D+1)), dtype=np.float64)
        #Strans[0] = S0
        xi = x0
        for i,(t1,t2) in enumerate(zip(ttrans[:-1], ttrans[1:])):
            xip1 = xi + RK4(f, xi, t1, t2, p)
            #Strans_temp = Strans[i] + RK4(dSdt, Strans[i], t1, t2, p)
            # perform QR decomposition on Phi
            #rPhi = np.reshape(Strans_temp[D:], (D, D))
            #Q,R = np.linalg.qr(rPhi)
            #Strans[i+1] = np.append(Strans_temp[:D], Q.flatten())
            xi = xip1
        x0 = xi

        #S0 = np.append(Strans[-1, :D], Phi0)
        #S0 = Strans[-1]

    # start LE calculation
    LE = np.zeros((N-1, D), dtype=np.float64)
    Ssol = np.zeros((N, D*(D+1)), dtype=np.float64)
    #Ssol[0] = S0
    Ssol[0] = np.append(x0, Phi0)

    print("Integrating system for LE calculation...")
    for i,(t1,t2) in enumerate(zip(t[:-1], t[1:])):
        dt = t2 - t1
        Ssol_temp = dapper.rk4(dSdt, Ssol[i], t1, dt)
        # perform QR decomposition on Phi
        rPhi = np.reshape(Ssol_temp[D:], (D, D))
        Q,R = np.linalg.qr(rPhi)
        Ssol[i+1] = np.append(Ssol_temp[:D], Q.flatten())
        LE[i] = np.abs(np.diag(R))
    return LE

def Compute_lyapunov_exponent(args):
    nx, F, G, alpha, gamma = args[0], args[1], args[2], args[3], args[4]
    spin_up_time = 100
    dt = 0.05
    nt = int(spin_up_time/dt)
    sd0 = dapper.seed_init(8)
    dapper.seed(sd0)
    x0, VL = simulate(np.random.rand(2*nx), nx, dt, nt, F, G, sd0, alpha = alpha, gamma = gamma)
    x0 = x0[-1, :]
    f = lambda t,x : VL.dxdt(x)
    fjac = lambda t,x: VL.d2x_dtdx(x)
    t = np.arange(0., 10000.05, 0.05)
    LEs = computeLE(f, fjac, x0, t, p=(), ttrans=None)

    return F, G, alpha, gamma, np.mean(np.log(LEs), axis = 0) /dt

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
    return xx, VL

def setup_LE_multi(alphas, gammas):
    configs = []
    for F, G in zip([10, 10, 0], [0, 10, 10]):
        for alpha in alphas:
            for gamma in gammas:
                configs +=[list([18, F, G, alpha, gamma])]

    import multiprocessing
    with multiprocessing.Pool(10) as p:
        for F, G, alpha, gamma, LEs in p.imap_unordered(Compute_lyapunov_exponent, configs):
            filename = f'LE_F{F}_G{G}_alpha{np.round(alpha, 3)}_gamma{np.round(gamma, 3)}.npy'
            np.save(filename, LEs)

if __name__ == "__main__":
    setup_LE_multi(alphas=np.arange(0., 3., 0.1), gammas=np.arange(0.3, 1.8, 0.05))
