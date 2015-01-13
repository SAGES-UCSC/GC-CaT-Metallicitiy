#!/usr/bin/python

import numpy as np
from scipy.optimize import leastsq, curve_fit
from scipy.special import gamma, psi

#fits xdata and ydata with a polynomial of order
#likely a better python way to generate the function but this works
def polyfit(order, xdata, ydata, p0=None, sigma=None, **kw):

	fstr = 'def f(x'
	statement = ', a1, a0): return '
	for i in range(order, 1, -1):
		fstr += ', a' + str(i)
		statement += 'x**' + str(i) + ' * a' + str(i) + ' + '  

	fstr += statement + 'x * a1 + a0'

	exec fstr

	return curve_fit(f, np.asarray(xdata), np.asarray(ydata), p0, sigma, **kw)	

#fit a power law
def powerfit(xdata, ydata, p0=None, sigma=None, **kw):

	def f(x, a, b):
		return b * x**a
		
	return curve_fit(f, np.asarray(xdata), np.asarray(ydata), p0, sigma, **kw)		

# fit a sersic profile
# still need some work
#def sersicfit(xdata, ydata, p0=None, sigma=None, **kw):
#
#	def f(x, ue, re, n):
#		return ue + 8.3268 * ( (x / re)**(1./n) - 1)
#
#	return curve_fit(f, xdata, ydata, p0, sigma, **kw)

def backsersicfit(xdata, ydata, p0=None, sigma=None, **kw):

	return curve_fit(backsersic, np.asarray(xdata), np.asarray(ydata), p0, sigma, **kw)

def backsersic(x, Ne, re, n, bg):
	
	return Ne * np.exp((-1.9992 * n + .3271) * ((x / re)**(1/n) - 1)) + bg

def sersicfit(xdata, ydata, p0=None, sigma=None, **kw):

	return curve_fit(sersic, np.asarray(xdata), np.asarray(ydata), p0, sigma, **kw)

def sersic(x, Ne, re, n, bg=0):
	return Ne * np.exp((-1.9992 * n + .3271) * ((x / re)**(1/n) - 1))

def intsersic(Ne, re, n, bg=0):
	bn = 1.9992 * n - .3271
	return Ne * re**2 * 2 * np.pi * n * np.e**bn / bn**(2 * n) * gamma(2 * n)

def intsersicerr(result):
	#print result
	Ne = result[0][0]
	re = result[0][1]
	n = result[0][2]
	bn = 1.9992 * n - .3271

	N = intsersic(Ne, re, n)

	d2Ne = result[1][0,0]
	d2re = result[1][1,1]
	d2n = result[1][2,2]
	dNedre = result[1][0,1]
	dNedn = result[1][0,2]
	dredn = result[1][1,2]

	#print Ne, re, n
	#print d2Ne, d2re, d2n, dNedre, dNedn, dredn

	dNdNe = N / Ne
	dNdre = 2 * N / re

	dNdn = N * (1 / n + 1.9992 + (-2 * np.log(bn) - 2 * n * 1.9992 / bn) + 2 * psi(2 * n))

	#print dNdNe, dNdre, dNdn

	return (dNdNe**2 * d2Ne + dNdre**2 * d2re + dNdn**2 * d2n + 2 * abs(dNdNe * dNdre) * dNedre + 2 * abs(dNdNe * dNdn) * dNedn + 2 * abs(dNdre * dNdn) * dredn)**.5

def bilinear(xdata, ydata, p0=None, sigma=None, **kw):
	
	def f(x, a1, a0):
		return x * a1 + a0
	
	firstfit = curve_fit(f,np.asarray(xdata), np.asarray(ydata), p0, sigma, **kw)
	secondfit = curve_fit(f, np.asarray(ydata), np.asarray(xdata), p0, sigma, **kw)
	
	print firstfit
	print secondfit
	
	xy = ((firstfit[0][0] + 1 / secondfit[0][0]) / 2, (firstfit[0][1] + -secondfit[0][1] / secondfit[0][0]) / 2 )
	dxy = ((firstfit[1][0,0] / 4 + secondfit[1][0,0] / (4 * secondfit[0][0]**4))**.5, (firstfit[1][1,1] / 4 + secondfit[1][1,1] / (4* secondfit[0][0]**2) + secondfit[0][1]**2 * secondfit[1][0,0] / (4 * secondfit[0][0]**4))**.5)
	
	yx = ((secondfit[0][0] + 1 / firstfit[0][0]) / 2, (secondfit[0][1] + -firstfit[0][1] / firstfit[0][0]) / 2 )
	dyx = ((secondfit[1][0,0] / 4 + firstfit[1][0,0] / (4 * firstfit[0][0]**4))**.5, (secondfit[1][1,1] / 4 + firstfit[1][1,1] / (4* firstfit[0][0]**2) + firstfit[0][1]**2 * firstfit[1][0,0] / (4 * firstfit[0][0]**4))**.5)
	
	return (xy, dxy, yx, dyx)

def broken(x, a, b, c, d):
	x = np.asarray(x)
	above = b * x + c
	below = d * x + c + (b - d) * a
	if x.shape:
		cut = x > a
		np.putmask(below, cut, above)
	else:
		if x > a:
			below = above
	return below

def brokenlinear(xdata, ydata, p0=None, sigma=None, **kw):
	
	return curve_fit(broken, np.asarray(xdata), np.asarray(ydata), p0, sigma, **kw)	
		

def test():
	import matplotlib.pyplot as plt

	x = np.arange(-1, 1, .01)

	y = np.poly1d([4, 3, -1])(x)

	plt.plot(x,y)

	plt.show()

	print polyfit(2, x, y)


if __name__ == "__main__":
	test()
