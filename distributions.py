import numpy as np
from scipy.special import gamma, erfinv
import scipy.integrate as integrate

def log_unnorm_normal(mean, variance):
    """Computes unnormalized log(f(x))=h(x) and h'(x) with f(x) being the pdf for the normal distribution.
    Args:
        mean: Specified mean of normal distribution
        variance: Specified variance of normal distribution
    Returns:
    h(x), h'(x)
    """
    return lambda x : (- 0.5 * (x - mean) ** 2 / variance, - (x - mean) / variance)

def log_norm_normal(mean, variance):
    """Computes normalized log(f(x))=h(x) and h'(x) with f(x) being the pdf for the normal distribution.
    Args:
        mean: Specified mean of normal distribution
        variance: Specified variance of normal distribution
    Returns:
    h(x), h'(x)
    """
    return lambda x : (np.log(1 / (np.sqrt(2 * np.pi * variance))) - 0.5 * (x - mean) ** 2 / variance, - (x - mean) / variance)

def log_unnorm_gamma(a, theta):
    """Computes unnormalized log(f(x))=h(x) and h'(x) with f(x) being the pdf for the gamma distribution.
    Args:
        a: Shape parameter in gamma distribution
        theta: Scale parameter in gamma distribution
    Returns:
    h(x), h'(x)
    """
    return lambda x : (np.log(x**(a-1)) - (x / theta), ((a - 1) / x) - (1 / theta))

def log_unnorm_fattailed(x):
    """Computes unnormalized log(f(x))=h(x) and h'(x) with f(x) being the pdf for the fat-tailed distribution.
    Args:
        a: Shape parameter in gamma distribution
        theta: Scale parameter in gamma distribution
    Returns:
    h(x), h'(x)
    """
    if isinstance(x, float):
        if x >= 1:
            return (np.log(np.exp(-(x-1)/2) + np.exp(-(x-1)**2)), -((4 * np.exp(x / 2) * (x-1) + np.exp((x - 1)**2 + 0.5)) / (2 * np.exp((x - 1)**2 + 0.5) + 2 * np.exp(x / 2))))
        else: 
            return (np.log(np.exp((x-1)/20) + np.exp((x-1)**3)), (3 * np.exp((x-1)**3) * (x-1)**2 + (1 / 20) * np.exp((x-1) / 20)) / (np.exp((x-1) / 20) + np.exp((x-1)**3)))

    h = x.copy()
    dhdx = x.copy() 
    h[x >= 1], dhdx[x >= 1] = np.log(np.exp(-(x[x>=1]-1)/2) + np.exp(-(x[x>=1]-1)**2)), - ((4 * np.exp(x[x>=1] / 2) * (x[x>=1]-1) + np.exp((x[x>=1] - 1)**2 + 0.5)) / (2 * np.exp((x[x>=1] - 1)**2 + 0.5) + 2 * np.exp(x[x>=1] / 2)))
    h[x < 1], dhdx[x < 1] = np.log(np.exp((x[x < 1]-1)/20) + np.exp((x[x < 1]-1)**3)), (3 * np.exp((x[x < 1]-1)**3) * (x[x < 1]-1)**2 + (1 / 20) * np.exp((x[x < 1]-1) / 20)) / (np.exp((x[x < 1]-1) / 20) + np.exp((x[x < 1]-1)**3))
    return h, dhdx

def log_unnorm_bimodal():
    """Computes unnormalized log(f(x))=h(x) and h'(x) with f(x) being the pdf for the gamma distribution.
    Args:
        a: Shape parameter in gamma distribution
        theta: Scale parameter in gamma distribution
    Returns:
    h(x), h'(x)
    """
    return lambda x : (np.log(0.3 * np.exp(-0.2 * x**2) + 0.7 * np.exp(-0.2 * (x - 10)**2)), (np.exp(0.2 * x**2) * (9.33333 - 0.933333 * x) - 0.4 * np.exp(0.2 * (x - 10)**2) * x) / (2.33333 * np.exp(0.2 * x**2)+ np.exp(0.2 * (x - 10)**2)))

def log_unnorm_multimodal():
    """Computes unnormalized log(f(x))=h(x) and h'(x) with f(x) being the pdf for the gamma distribution.
    Args:
        a: Shape parameter in gamma distribution
        theta: Scale parameter in gamma distribution
    Returns:
    h(x), h'(x)
    """
    return lambda x : ((-x**2 / 2) + np.log((np.sin(0.2 * x)**2 + 4 * np.cos(0.01 * x)**2 * np.sin(6 * x)**2 + 8)), (0.2 * np.sin(0.4 * x) + np.sin(6 * x) *(48 * np.cos(0.01 * x)**2 * np.cos(6 * x) - 0.04 * np.sin(0.02 * x) * np.sin(6 * x))) / (np.sin(0.2 * x)**2 + 4 * np.sin(6* x)**2 * np.cos(0.01 * x)**2 + 8) - x)


class Logistic:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        return np.exp(-(x - self.loc) / self.scale) / (self.scale * (1 + np.exp(-(x - self.loc) / self.scale))**2)

    def rvs(self, N):
        x = np.random.uniform(low=0, high=1, size=N)
        return -(self.scale * np.log(1 / x - 1) - self.loc)


class Gamma:
    def __init__(self, a, beta):
        self.a = a
        self.beta = beta

    def pdf(self, x):
        if isinstance(x, float):
            if x >= 0:
                return (x**(self.a - 1) * np.exp(-self.beta * x) * self.beta**self.a) / gamma(self.a)
            else:
                return 0
        y = x.copy()
        y[y >= 0] = (y[y >= 0]**(self.a - 1) * np.exp(-self.beta * y[y >= 0]) * self.beta**self.a) / gamma(self.a)
        y[y < 0] = 0
        return y

    def norm_const(self):
        return 1


class Cauchy:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        return (1 / (np.pi * (1 + ((x - self.loc) / self.scale)**2))) / self.scale

    def rvs(self, N):
        x = np.random.uniform(low=0, high=1, size=N)
        return self.scale * np.tan(np.pi * (x - 1 /2)) + self.loc

class Normal:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        return (1 / (self.scale * np.sqrt(2 * np.pi))) * np.exp(- 0.5 * ((x - self.loc)**2 / self.scale**2))

    def rvs(self, N):
        x = np.random.uniform(low=0, high=1, size=N)
        return self.scale * np.sqrt(2) * erfinv(2 * x - 1) + self.loc



class fat_tailed_distribution:

    def pdf(self, x):

        NORM_CONST = 1 / 24.040550419

        if isinstance(x, float):
            if x >= 1:
                return NORM_CONST * (np.exp(-(x-1)/2) + np.exp(-(x-1)**2))
            else: 
                return NORM_CONST * (np.exp((x-1)/20) + np.exp((x-1)**3))
        y = x.copy()
        y[x >= 1] = NORM_CONST * (np.exp(-(x[x>=1]-1)/2) + np.exp(-(x[x>=1]-1)**2))
        y[x < 1] = NORM_CONST * (np.exp((x[x < 1]-1)/20) + np.exp((x[x < 1]-1)**3))
        return y

    def norm_const(self):
        return 1 / 24.040550419

class fat_tailed_distribution_unnormalized:

    def pdf(self, x):

        if isinstance(x, float):
            if x >= 1:
                return (np.exp(-(x-1)/2) + np.exp(-(x-1)**2))
            else: 
                return (np.exp((x-1)/20) + np.exp((x-1)**3))
        y = x.copy()
        y[x >= 1] = (np.exp(-(x[x>=1]-1)/2) + np.exp(-(x[x>=1]-1)**2))
        y[x < 1] = (np.exp((x[x < 1]-1)/20) + np.exp((x[x < 1]-1)**3))
        return y

    def norm_const(self):
        return 1 / 24.040550419

class unnormalized_bimodal_distribution:

    def pdf(self, x):
        return 0.3 * np.exp(-0.2 * x**2) + 0.7 * np.exp(-0.2 * (x - 10)**2)

    def norm_const(self):
        return 1 / integrate.quad(lambda x: 0.3 * np.exp(-0.2 * x**2) + 0.7 * np.exp(-0.2 * (x - 10)**2), -np.inf, np.inf)[0]

class normalized_bimodal_distribution:

    def pdf(self, x):
        return self.norm_const() * ( 0.3 * np.exp(-0.2 * x**2) + 0.7 * np.exp(-0.2 * (x - 10)**2))

    def norm_const(self):
        return 1 / integrate.quad(lambda x: 0.3 * np.exp(-0.2 * x**2) + 0.7 * np.exp(-0.2 * (x - 10)**2), -np.inf, np.inf)[0]

class unnormalized_multimodal_distribution:

    def pdf(self, x):
        return np.exp(-x**2 / 2) * (np.sin(0.2 * x)**2 + 4 * np.cos(0.01 * x)**2 * np.sin(6 * x)**2 + 8)

    def norm_const(self):
        return 1 / 25.1656

class normalized_multimodal_distribution:

    def pdf(self, x):
        return self.norm_const() * (np.exp(-x**2 / 2) * (np.sin(0.2 * x)**2 + 4 * np.cos(0.01 * x)**2 * np.sin(6 * x)**2 + 8))

    def norm_const(self):
        return 1 / 25.1656

class mixture_gaussian:
    def __init__(self, rv1, rv2, rv3):
        self.rv1 = rv1
        self.rv2 = rv2
        self.rv3 = rv3

    def pdf(self, x):
        return 1 * self.rv1.pdf(x) + 0.4 * self.rv2.pdf(x) + 0.6 * self.rv3.pdf(x)

class mixture_gaussian_5D:
    def __init__(self, rv1, rv2, rv3):
        self.rv1 = rv1
        self.rv2 = rv2
        self.rv3 = rv3

    def pdf(self, x):
        return (1 / 3) * self.rv1.pdf(x) + (1 / 3) * self.rv2.pdf(x) + (1 / 3) * self.rv3.pdf(x)
