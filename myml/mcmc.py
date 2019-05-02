import numpy as np


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
        self.w = x

    def set_q(self, cov):
        self.qcov = cov
        if self.ndim == 1:
            self.qtra = cov
        else:
            self.qtra = np.linalg.cholesky(cov)

    def draw_q(self, n):
        if self.qtra:
            norm = np.random.randn(n, self.ndim)
            return norm@self.qtra.T

    def propose_step(self):
        add = self.draw_q(self.nwalkers)
        return self.w + add

    def ratio(self, x):
        #if self.ndim == 1:
        #    ratios = np.zeros(x.size, dtype=np.float64)
        #    for i in range(x.size):
        #        ratios[i] = self.lnf(x[i]) - self.lnf(self.w[i])
        #    return ratios
        ratios = np.zeros(x.shape[0], dtype=np.float64)
        for i in range(x.shape[0]):
            ratios[i] = self.lnf(x[i, :]) - self.lnf(self.w[i, :])
        return ratios

    def step(self, n):
        #if self.ndim == 1:
        #    chains = np.zeros((n, self.nwalkers), dtype=np.float64)
        #    for i in range(n):
        #        proposed = self.propose_step()
        #        Q = self.ratio(proposed)
        #        R = np.random.rand(self.nwalkers)
        #        for j in range(self.nwalkers):
        #            if Q[j] > R[j]:
        #                self.w[j] = proposed[j]
        #        chains[i, :] += self.w
        #    return np.reshape(chains, (n*self.nwalkers)), chains
        chains = np.zeros((n, self.nwalkers, self.ndim), dtype=np.float64)
        for i in range(n):
            proposed = self.propose_step()
            Q = self.ratio(proposed)
            R = np.random.rand(self.nwalkers)
            for j in range(self.nwalkers):
                if Q[j] > R[j]:
                    self.w[j, :] = proposed[j, :]
            chains[i, :, :] += self.w
        return np.reshape(chains, (n*self.nwalkers, self.ndim)), chains
