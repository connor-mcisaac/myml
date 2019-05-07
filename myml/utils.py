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
