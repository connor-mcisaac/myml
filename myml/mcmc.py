import numpy as np
from scipy.interpolate import interp1d


class draw1D(object):

    def __init__(self, function, lower, upper, args=None):
        xs = np.linspace(lower, upper, num=1000, endpoint=True)
        if args is None:
            ps = function(xs)
        else:
            ps = function(xs, *args)
        cs = np.cumsum(ps)
        cs /= cs[-1]
        self.drawer = interp1d(cs, xs, bounds_error=False,
                               fill_value='extrapolate')

    def __call__(self, n):
        u = np.random.rand(n)
        draws = self.drawer(u)
        return draws


class mymetropolis(object):


    def __init__(self, ndim, nwalkers, lnp, lnl):
        self.ndim = ndim
        self.nwalkers = nwalkers
        self.lnp = lnp
        self.lnl = lnl
        self.w = np.random.randn(self.nwalkers, self.ndim)

    def lnf(self, x):
        p = self.lnp(x)
        if not np.isfinite(p):
            return -np.inf
        l = self.lnl(x)
        if not np.isfinite(l):
            return -np.inf
        return p + l

    def set_initial(self, x):
        if x.shape != self.w.shape:
            err_msg = "Shape of initial value doesn't match required shape"
            err_msg += '-> {}'.format(self.w.shape)
            raise ValueError(err_msg)
        self.w = x

    def set_q(self, q, lower=-1, upper=1):
        if callable(q):
            self.qtra = draw1D(q, lower, upper)
        elif type(q) == np.ndarray:
            self.qcov = q
            if self.ndim == 1:
                self.qtra = q
            else:
                self.qtra = np.linalg.cholesky(q)

    def draw_q(self, n):
        if not callable(self.qtra):
            norm = np.random.randn(n, self.ndim)
            draws = norm@self.qtra.T
        else:
            draws = np.zeros((n, self.ndim))
            for i in range(n):
                draws[i, :] = self.qtra(self.ndim)
        if self.ndim == 1:
            draws = np.reshape(draws, (-1, 1))
        return draws

    def propose_step(self):
        add = self.draw_q(self.nwalkers)
        return self.w + add

    def ratio(self, x):
        ratios = np.zeros(x.shape[0], dtype=np.float64)
        for i in range(x.shape[0]):
            ratios[i] = self.lnf(x[i, :]) - self.lnf(self.w[i, :])
        return ratios

    def step(self, n):
        chains = np.zeros((n, self.nwalkers, self.ndim), dtype=np.float64)
        for i in range(n):
            proposed = self.propose_step()
            Q = self.ratio(proposed)
            R = np.random.rand(self.nwalkers)
            for j in range(self.nwalkers):
                if Q[j] > np.log(R[j]):
                    self.w[j, :] = proposed[j, :]
            chains[i, :, :] += self.w
        return np.reshape(chains, (n*self.nwalkers, self.ndim)), chains
