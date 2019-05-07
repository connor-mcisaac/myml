import numpy
from scipy.interpolate import interp1d


def create_diag(n):
    d = numpy.zeros((n**n), dtype=numpy.float64)
    j = numpy.sum([n**i for i in range(n)])
    for i in range(n):
        d[i*j] += 1
    return d.reshape([n for i in range(n)])


def create_diag2D(n):
    d = numpy.zeros((n, n), dtype=numpy.float64)
    for i in range(n):
        d[i, i] += 1
    return d


class draw1D(object):

    def __init__(self, function, lower, upper, args=None):
        xs = numpy.linspace(lower, upper, num=1000, endpoint=True)
        ps = numpy.zeros((1000), dtype=numpy.float64)
        if args is None:
            for i in range(1000):
                ps[i] += function(xs[i])
        else:
            for i in range(1000):
                ps[i] += function(xs[i], *args)
        cs = numpy.cumsum(ps)
        cs /= cs[-1]
        self.drawer = interp1d(cs, xs, bounds_error=False,
                               fill_value='extrapolate')

    def __call__(self, n):
        u = numpy.random.rand(n)
        draws = self.drawer(u)
        return draws


class mn(object):

    def __init__(self, m, c):
        self.dim = numpy.size(m)
        self.m = m
        self.c = c
        if self.dim == 1:
            self.ic = 1/c
            self.dc = c
        else:
            self.ic = numpy.linalg.inv(c)
            self.dc = numpy.linalg.det(c)

    def __call__(self, x):
        if self.dim == 1:
            return (numpy.exp(-0.5*(x-self.m)*self.ic*(x-self.m))
                    /numpy.sqrt((((2*numpy.pi)**self.dim)*self.dc)))
        else:
            return (numpy.exp(-0.5*(x-self.m).T@self.ic@(x-self.m))
                    /numpy.sqrt((((2*numpy.pi)**self.dim)*self.dc)))
