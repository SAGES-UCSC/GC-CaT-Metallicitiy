#! /usr/bin/env python 
'''
Created on Sep 24, 2012

@author: Chris Usher
'''

import numpy as np
import scipy.stats as stats
import curvefit
import subprocess
import scipy.interpolate as interpolate


def rms(xs, ys):
    return ((xs - ys)**2).mean()**0.5

def chi2(xs, ys, params, yes, xes=0):
    chi2 = ((ys - xs)**2 / (xes**2 + yes**2)).sum()
    dof = max(xs.size, ys.size) - params
    p = 1 - stats.chi2.cdf(chi2, dof)
    return chi2, dof, p

def chi2_asym(xs, ys, params, xles, xues, yles, yues):
    
    above = (ys - xs)**2 / (yles**2 + xues**2)
    below = (ys - xs)**2 / (yues**2 + xles**2)
    cut = ys > xs
    np.putmask(below, cut, above)
    dof = below.size - params
    chi2 = below.sum()
    p = 1 - stats.chi2.cdf(chi2, dof)
    return chi2, dof, p

def akaike(c2, params, size):
    
    return c2 + 2 * params + (2. * params * (params + 1)) / (size - params - 1.)


def bootstrap(xs, ys, yes, func, n=1000, names=None, title='', verbose=True, plot=True, p0=None, fitting_func=None, optimizer=None, counter=False):

    np.random.seed(0)

    if verbose:
        print '\nFitting \'' + title + '\'\n---- ----'
        
    if fitting_func==None:
        fitting_func = func
        
    
    xs = np.asarray(xs)
    
    if optimizer == None:
        fit = curvefit.curve_fit(fitting_func, xs, ys, p0=p0, sigma=yes)[0]
    
        def newfs(x):
            return func(x, *fit)
    
        c2 = chi2(newfs(xs), ys, fit.size, yes)
    
        mean, std = wmean(ys, yes)
        const_c2 = chi2(mean, ys, 1, yes) 
    
        if verbose:
            print 'Mean:', mean, 'STD:', std
            print 'chi2:', c2[0], 'Dof:', c2[1], 'p:', c2[2]
            print 'RMS:', ((newfs(xs) - ys)**2).mean()**.5
            print 'Akaike:', akaike(c2[0], fit.size, xs.size)
            spearman = stats.spearmanr(newfs(xs), ys)
            print 'Spearman rank correlation:', spearman[0], 'p:', spearman[1]
            print 'Constant chi2:', const_c2[0], 'Dof:', const_c2[1], 'p:', const_c2[2]
            print 'Constant RMS:', ((mean - ys)**2).mean()**.5
            print 'Constant Akaike:', akaike(const_c2[0], 1, xs.size)
            print

    else:
        fit = optimizer(xs, ys, yes)
        newfs = func(*fit)
        c2 = None
        
    fits = []
    i = 0
    
    length = xs.shape[-1]
    while i < n:
        try:
            points = np.random.random_integers(0, length - 1, length)
            sample_xs = xs.T[points].T
            sample_ys = ys[points]
            sample_yes = yes[points] 
            
            if optimizer == None:
                sample_fit = curvefit.curve_fit(fitting_func, sample_xs, sample_ys, p0=p0, sigma=sample_yes)[0]
            else:
                sample_fit = optimizer(sample_xs, sample_ys, sample_yes)
            
            fits.append(sample_fit)
            i += 1
            if counter:
                print i, 'of', n
            
        except(RuntimeError, subprocess.CalledProcessError):
            print 'Error'
            pass

    fits = np.array(fits)
    if names == None or len(names) != fits.shape[1]:
        names = range(fits.shape[1])
    
    lowers = []
    uppers = []
    for i in range(fits.shape[1]):
        
        fits = fits[fits[:,i].argsort()]
        
        lower = fits[int(.16 * n), i] - fit[i]
        upper = fits[int(.84 * n) - 1, i] - fit[i]
        lowers.append(lower)
        uppers.append(upper)         
        
        above_zero = fits[fits[:,i] > 0].shape[0] / float(fits.shape[0])
        
        if plot:
            import matplotlib.pyplot as plt
            plt.figure()
            
            
            binwidth = (upper - lower) / 10
            nbins = int((fits[:,i].max() - fits[:,i].min()) / binwidth + 1)
            
            plt.hist(fits[:,i], histtype='step', bins=nbins)
            plt.title(title + ' ' + str(names[i]))

        if verbose:
            print str(names[i]) + ':'
            
            print fit[i], lower, upper
            print 'p(>0):', above_zero 



          
#        null result chi
            
    return newfs, fit, lowers, uppers, c2, fits


def wmean(xs, es):
    ivars = es**-2
    mean = (xs * ivars).sum() / ivars.sum()
    std = ivars.sum()**-.5
    
    return mean, std
    

def poly(order):
    
    def f():
        pass
    
    fstr = 'def f(x'
    statement = ', a1, a0): return '
    for i in range(order, 1, -1):
        fstr += ', a' + str(i)
        statement += 'x**' + str(i) + ' * a' + str(i) + ' + '  

    fstr += statement + 'x * a1 + a0'

    exec fstr
    
    return f

def power(x, a, b):
    return 10**b * x**a

def expo(x, a, b):
    return 10**(a * x + b)
        
def unbiased_broken(x, a, b, c, d):
    
#    print a, x.min(), x.max()
#    if a < x.min() or a > x.min():
    if a < 0:
        return 1e100
    return curvefit.broken(x, a, b, c, d)    
        
def twolinear(x, a, b, k):
    
    return a * x[0] + b * x[1] + k

def threelinear(x, a, b, c, k):
    
    return a * x[0] + b * x[1] + c * x[2] + k            
    
def spearman(xs, ys, n=1e4, plot=False):
    rho, p = stats.spearmanr(xs, ys)
    
    print rho, p
    
    xs = np.asarray(xs)
    rhos = []
    
    for i in range(int(n)):
        permutation = np.random.permutation(xs)
        
        rhos.append(stats.spearmanr(permutation, ys)[0])
        
    if plot:
        import matplotlib.pyplot as plt    
        plt.figure()
        plt.hist(rhos, bins=100, range=(-1, 1), histtype='step')
    
    rhos = np.array(rhos)
    
    if rho > 0:
        print rhos[rhos > rho].size / float(n)
    else:
        print rhos[rhos < rho].size / float(n)

# calculate n sigma interval of data
# start at peak probability
# integrate between two points where pdf function equals half max prob

def kde_interval(data, sigma=1):
    
    p_sigma = stats.norm.cdf(sigma) - stats.norm.cdf(-sigma)
#    print p_sigma
    data = np.asanyarray(data)
    
    xs = np.linspace(2 * data.min() - data.mean(), 2 * data.max() - data.mean(), 512)
    
    # use gaussian kernel estimation and a spline to create an analytic pdf from data array
    spline = interpolate.InterpolatedUnivariateSpline(xs, stats.gaussian_kde(data)(xs))
    
    pdf_max = spline(xs).max()
    pdf = pdf_max / 2
    count = -2
    diff = 1
    roots = None
    
    while diff > 0.0025 and count > -20:
        
#        print pdf_max, pdf, roots
        
        shifted_spline = interpolate.InterpolatedUnivariateSpline(xs, spline(xs) - pdf)
        roots = shifted_spline.roots()
    
        if roots.size % 2 != 0:
            roots = None
            pdf += 2**-20
            continue
        
#            p = 0
#            for j in range(roots.size / 2):
#            
#                p += spline.integral(roots[j * 2], roots[j * 2 + 1])

        p = spline.integral(roots[0], roots[-1])
            
#        print pdf, roots, p            
        
        if p > p_sigma:
            pdf += pdf_max * 2**count
        else:
            pdf -= pdf_max * 2**count
        
        count += -1
        diff = np.abs(p - sigma)    
    
    peak = xs[spline(xs) == pdf_max].mean()
    
    sorted_data = data[data.argsort()]

#    print sorted_data[stats.norm.cdf(-sigma) * data.size], sorted_data[stats.norm.cdf(sigma) * data.size]
    
    return (peak, roots[0], roots[-1], spline)
        