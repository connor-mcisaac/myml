import numpy
from myml.utils import create_diag2D, draw1D, mn


class mymetropolis(object):

    def __init__(self, ndim, nwalkers, lnp, lnl, pargs=[], largs=[]):
        self.ndim = ndim
        self.nwalkers = nwalkers
        self.lnp = lnp
        self.lnl = lnl
        self.pargs = pargs
        self.largs = largs
        self.w = numpy.random.randn(self.nwalkers, self.ndim)
        self.set_q()

    def lnf(self, x):
        p = self.lnp(x, *self.pargs)
        if not numpy.isfinite(p):
            return -numpy.inf
        l = self.lnl(x, *self.largs)
        if not numpy.isfinite(l):
            return -numpy.inf
        return p + l

    def set_initial(self, x):
        if x.shape != self.w.shape:
            err_msg = "Shape of initial value doesn't match required shape"
            err_msg += '-> {}'.format(self.w.shape)
            raise ValueError(err_msg)
        self.w = x

    def set_q(self, q=None, lower=-1, upper=1, c=None):
        if c is None:
            self.qcov = create_diag2D(self.ndim)
        elif type(c) == numpy.ndarray:
            self.qcov = c
        if self.ndim == 1:
            self.qtra = numpy.sqrt(self.qcov)
        else:
            self.qtra = numpy.linalg.cholesky(self.qcov)
        if q is None:
            self.q = mn(numpy.zeros((self.ndim)), create_diag2D(self.ndim))
            self.qdraw = numpy.random.randn
        elif callable(q):
            self.q = q
            self.qdraw = draw1D(q, lower, upper)

    def draw_q(self, n):
        draws = numpy.zeros((n, self.ndim))
        for i in range(self.ndim):
            draws[:, i] += self.qdraw(n)
        if self.ndim == 1:
            draws = numpy.reshape(draws, (-1, 1))*self.qtra
        else:
            draws = draws@self.qtra.T
        return draws

    def propose_step(self):
        add = self.draw_q(self.nwalkers)
        return self.w + add

    def ratio(self, x):
        ratios = numpy.zeros(x.shape[0], dtype=numpy.float64)
        for i in range(x.shape[0]):
            ratios[i] = self.lnf(x[i, :]) - self.lnf(self.w[i, :])
        return ratios

    def step(self, n):
        chains = numpy.zeros((n, self.nwalkers, self.ndim), dtype=numpy.float64)
        tried = numpy.zeros((self.nwalkers), dtype=numpy.int32)
        passed = numpy.zeros((self.nwalkers), dtype=numpy.int32)
        for i in range(n):
            proposed = self.propose_step()
            Q = self.ratio(proposed)
            R = numpy.random.rand(self.nwalkers)
            for j in range(self.nwalkers):
                tried[j] += 1
                if Q[j] > numpy.log(R[j]):
                    passed[j] += 1
                    self.w[j, :] = proposed[j, :]
            chains[i, :, :] += self.w
        print('Run complete, acceptance fraction = {}'
              .format(numpy.sum(passed)/numpy.sum(tried)))
        return numpy.reshape(chains, (n*self.nwalkers, self.ndim)), chains


class mymh(mymetropolis):

    def __init__(self, ndim, nwalkers, lnp, lnl, pargs=[], largs=[]):
        super().__init__(ndim, nwalkers, lnp, lnl, pargs=pargs, largs=largs)

    def ratio(self, x):
        ratios = numpy.zeros(x.shape[0], dtype=numpy.float64)
        for i in range(x.shape[0]):
            qfactor = 0
            if self.ndim == 1:
                diff = (self.w[i, :] - x[i, :])*self.qtra
            else:
                diff = (self.w[i, :] - x[i, :])@self.qtra.T
            for j in range(self.ndim):
                qfactor += (numpy.log(self.q(diff[j]))
                            - numpy.log(self.q(-diff[j])))
            ratios[i] = self.lnf(x[i, :]) - self.lnf(self.w[i, :]) + qfactor
        return ratios
