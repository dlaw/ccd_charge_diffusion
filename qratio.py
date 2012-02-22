from numpy import *
from scipy.special import erf
import utils

def qratio_sim(points, sigma):
    p, s = points[...,None,:,None], sigma[...,None,None,None]
    return (divide.reduce(prod(subtract.reduce(sum(erf(divide(
        [[[2.,-1.]],[[1.,0.]]]-p,s*sqrt(2))),0),-1),-1),-1)-1)/8

def qratio(boxes):
    return (sum(boxes)/sum(boxes[...,1,1])-1)/8

def boxes(pixels, ka=None):
    t = utils.totals(pixels, n=3)
    m = utils.maxima(pixels, n=2)[...,1:-1,1:-1]
    ka = ka or utils.k_alpha(t[m])
    m &= (abs(t - ka) < .15*ka)
    return utils.boxes(m, pixels, n=3)
