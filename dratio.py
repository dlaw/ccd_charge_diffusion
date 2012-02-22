from numpy import *
from scipy.special import erf, erfinv
from scipy.optimize import fmin
from mykde import gaussian_kde
import utils

def dratio_sim(x, sigma):
    return erf(x / (sigma*sqrt(2)))

def dratio(boxes):
    pairs = concatenate([sum(boxes,1), sum(boxes,2)])
    return abs(subtract.reduce(pairs,1) / sum(pairs,1))

def sigma_peak(dratio, guess=1., cov=.0004):
    k = gaussian_kde(dratio, cov)
    peak = asscalar(fmin(lambda x: -k(x), guess, disp=0))
    return 1 / erfinv(peak) / sqrt(8)

def sigma_plateau(dratio, cov=.01):
    k = gaussian_kde(dratio, cov)
    result = asscalar(k(0)) * 2
    return result / sqrt(2*pi)

def boxes(pixels, ka=None):
    t = utils.totals(pixels)
    m = utils.maxima(t)
    ka = ka or utils.k_alpha(t[m])
    m &= (abs(t - ka) < .15*ka)
    return utils.boxes(m, pixels)

def analyze(pixels):
    d = dratio(boxes(pixels))
    return (sigma_plateau(d), sigma_peak(d))
