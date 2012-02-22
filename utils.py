import numpy
from itertools import product, izip

def totals(arr, n=2):
    """Find the sums of all n-by-n boxes in arr.  The return array is
    smaller than arr by (n-1) in the two final dimensions."""
    out = numpy.zeros_like(arr[..., n-1:, n-1:])
    s = [slice(i, 1+i-n or None) for i in range(n)]
    for x,y in product(s, repeat=2): out += arr[...,x,y]
    return out

def maxima(arr, n=2):
    """Find elements of arr that are larger than all other elements of
    arr within an n-by-n box.  Returns a boolean array with the same
    shape as arr."""
    out = numpy.ones(arr.shape, dtype='bool')
    s = [slice(max(i,0), min(i,0) or None) for i in range(1-n,n)]
    for (x,a),(y,b) in product(izip(s, reversed(s)), repeat=2):
        out[...,x,y] &= arr[...,a,b] <= arr[...,x,y]
    return out

def k_alpha(arr, lower=100, upper=500, steps=200):
    """Divide the range between lower and upper into steps sections
    and determine which section contains the k-alpha peak."""
    h = numpy.histogram(arr, steps, (lower,upper))
    return (h[1][:-1]+h[1][1:])[h[0].argmax()]/2

def boxes(locs, arr, n=2):
    """Extract n-by-n boxes from arr at the locations given by locs.
    Uses fancy indexing, so may not work for large datasets."""
    if (len(arr.shape) == 2): locs, arr = locs[None], arr[None]
    ix = numpy.array(numpy.where(locs))[...,None,None]
    return arr[tuple(ix + numpy.mgrid[:1,:n,:n])]
