import numpy as np

class model():
    """
    Use OOP to facilitate having multiple parameter settings simultaneously.
    """
    def __init__(self, nX, F, G, alpha=1, gamma=1, adv_control = 1):

        # System size
        self.nX = nX       # num of gridpoints
        self.M  = 2*nX     # => Total state length

        # Other parameters
        self.F  = F          # forcing for X
        self.G  = G          # forcing for theta
        self.alpha  = alpha  # energy conversion coefficient
        self.gamma  = gamma  # dissipation coefficient
        self.adv_control = adv_control
        # Init with perturbation
        self.x0 = np.zeros(self.M)#[0]

    def shift(self, x,n):
        return np.roll(x,-n,axis=-1)

    def unpack(self,x):
        X  = x[..., :self.nX]
        theta = x[..., self.nX:]
        return self.nX, self.F, self.G, self.alpha, self.gamma, self.adv_control, X, theta

    def dxdt(self,x):
        """Full (coupled) dxdt."""
        nX, F, G, alpha, gamma, adv_control, X, theta = self.unpack(x)
        d = np.zeros_like(x)
        d[...,:nX]  = adv_control*( self.shift(X,1)-self.shift(X,-2) )*self.shift(X,-1) - gamma*self.shift(X,0)
        d[...,:nX] += -alpha*self.shift(theta, 0) + F
        d[...,nX:]  = adv_control*(self.shift(X,1)*self.shift(theta,2)-self.shift(X,-1)*self.shift(theta,-2)) - gamma*self.shift(theta, 0)
        d[...,nX:] += alpha*self.shift(X, 0) + G

        return d

    def d2x_dtdx(self,x):
      nX, F, G, alpha, gamma, X, theta = self.unpack(x)
      md = lambda i: np.mod(i, nX) # modulo

      F = np.eye(self.M)
      F = F*(-gamma)
      for k in range(nX):
        # dX/dX
        F[k, md(k-1)]     = X[md(k+1)]-X[k-2]
        F[k, md(k+1)] = X[k-1]
        F[k, md(k-2)]     = -X[k-1]
        #dX/dtheta   
        F[k, k + nX] = -alpha

        #dtheta/dX
        F[k + nX, k+1] = theta[md(k+2)]
        F[k + nX, md(k-1)] = -theta[k-2]
        F[k + nX, k] = alpha
        #dtheta/dtheta
        F[k + nX, md(k +2)  + nX] = X[md(k+1)]
        F[k + nX, md(k -2)  + nX] = -X[k-1]

      return F

    def std(self, F, G):
        r"""
            Standard deviation of model variable X and $\theta$

            Parameters
            ----------
            label : str
                    A string consists of the size of gridpoints, forcing of X and $\theta$
                    e.g. '36100' represents 36 gridpoints, F = 10 and G = 0.
        """
        std = {}
        std['36100']  = np.array([3.693, 2.3])
        std['361010'] = np.array([3.531, 4.497])
        std['36010']  = np.array([2.508, 3.743])
        std['3688']   = np.array([2.867, 3.847])
        #
        std['40100']  = np.array([3.683, 2.308])
        std['401010'] = np.array([3.532, 4.494])
        std['40010']  = np.array([2.509, 3.738])
        std['4088']   = np.array([2.868, 3.842])
        #
        std['18100']  = np.array([3.723, 2.276])
        std['181010'] = np.array([3.528, 4.496])
        std['18010']  = np.array([2.509, 3.741])
        std['1888']   = np.array([2.866, 3.846])
        
        return std[label]
    # def dstep_dx(self,x,t,dt):
    #   return integrate_TLM(self.d2x_dtdx_auto(x),dt,method='analytic')

    # def LPs(self,jj):
    #   return [ ( 11, 1, LP.spatial1d(jj,dims=list(range(self.nU))) ) ]

