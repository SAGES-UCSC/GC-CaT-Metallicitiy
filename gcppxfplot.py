#! /usr/bin/env python


import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse
import scipy.constants as constants
import glob
import sys

parser = argparse.ArgumentParser(description='Plot GC pxf outputs')
parser.add_argument('-w', '--weak-lines', action='store_true', default=False, help='print weak line measurements')
parser.add_argument('pickles', nargs='*', help='pickles to examine')
parser.add_argument('-s', '--s2n-cut', action='store_true', default=False, help='skip over low signal to noise spectra')
args = parser.parse_args()

if len(args.pickles):
	pickles = args.pickles
else:
	pickles = glob.glob('*.pickle')
	if not pickles:
		print 'No pickle files found'
		sys.exit()

pickles.sort()
print 'Number of output pickles:', len(pickles)

transform = np.poly1d([0.46058111, -3.75039355])

def display(pickles, index, specific):
		try:
			data = pickle.load(open(pickles[index]))
	
		except Exception, er:
		
			print er, type(er)
			print pickles[index], 'is either not found or not a valid pickle file'
		else:
			if args.s2n_cut and not specific:
				if data.s2n < 8:
					print data.file, 'signal to noise too low (', data.s2n, ')'
					return False
				
				
			print '================================================================================'
			print 'Object:', data.id
			print 'File:', data.file
			print 'Runtime:', data.runtime
			print 'Signal to Noise:', data.s2n
			print 'CaT: %(value).3f-%(low).3f+%(high).3f' % {'value':data.CaT, 'low':data.CaTle, 'high':data.CaTue}
			print '[Z/H]: %(value).3f%(low).3f+%(high).3f' % {'value':transform(data.CaT), 'low':transform(data.CaT - data.CaTle) - transform(data.CaT), 'high':transform(data.CaT + data.CaTue) - transform(data.CaT)}
			if hasattr(data, 'fitrv'):
				print u'\u03C3: %(sigma).3f RV: %(rv).3f \u0394RV: %(drv).3f' % {'drv':data.rv - data.fitrv, 'sigma':data.fitsigma, 'rv':data.rv}
			else:
				print 'RV: %(rv).3f' % {'rv':data.rv}
			
			print 'Radius: %(value).1f arcsec' % {'value':data.r * 3600}			
			print index + 1, 'of', len(pickles)
			print '--------------------------------------------------------------------------------'

			if hasattr(data, 'fitrv'):
				zp1 = 1 + data.fitrv * 1e3 / constants.c
			else:
				zp1 = 1 + data.rv * 1e3 / constants.c
			
			fig = plt.figure(figsize=(16,10))
			fig.canvas.set_window_title(data.file)
			fig.subplots_adjust(left=.05, right=.95, bottom=.05, top=.95, wspace=.05, hspace=.15)
			
			plt.subplot(2,1,1)
			plt.plot(data.catwavelengths / zp1, data.catfluxes, 'b-', label='Input Spectra')
			plt.plot(data.fitwavelengths, data.fitfluxes, 'k-', label='Fit')
			plt.plot(data.samplewavelengths, data.lowsample, 'k:', label='Limits')
			plt.plot(data.samplewavelengths, data.highsample, 'k:', label=None)
			plt.hlines(np.ones(data.restframemask.shape[0]) * data.catfluxes.mean(), data.restframemask[:,0], data.restframemask[:,1], color='r', label='Masked Pixels')
			plt.plot(data.catwavelengths / zp1, data.catsigmas, 'g-', label='Error Array')
			plt.legend(ncol=3, frameon=False, loc=3)
			plt.xlabel('Wavelength')
			plt.ylabel('Flux')
			plt.title('Fitted Spectra: ' + data.file)
			plt.xlim(8380, 8920)
			ymin = max(-data.catfluxes.std(), -10)
			ymax = data.catfluxes.mean() + 2 * data.catfluxes.std()
			
			plt.ylim(ymin, ymax)
			
			plt.subplot(2,1,2)
			plt.plot(data.fitwavelengths, data.normfluxes, 'k-', label='Normalised Fit')
			plt.plot(data.samplewavelengths, data.lownormsample, 'k:', label='Limits')
			plt.plot(data.samplewavelengths, data.highnormsample, 'k:', label=None)
			plt.hlines(np.ones(data.restframemask.shape[0]), data.restframemask[:,0], data.restframemask[:,1], color='r', label='Masked Pixels')
			plt.legend(ncol=2, frameon=False, loc=3)
			plt.xlabel('Wavelength')
			plt.ylabel('Normalised Flux')
			plt.title('Normalised Spectra: ' + data.file)
			plt.xlim(8380, 8920)
			plt.ylim(.25, 1.15)
			
			plt.show()
			return True

try:
	index = 0
	display(pickles, index, False)
	
	while True:
		key = raw_input()
		specific = False
		if key == 'q':
			break
		elif key in ['b', 'p']:
			index += -1
		elif key in ['n', '']:
			index += 1
			
		elif key in ['h', 'help']:
			print 'Key commands for gcppxfplot'
			print 'q : Quit'
			print 'n or \'\' : Next spectra'
			print 'b or p : Previous spepctra'
			print 'Any other text : Search for pickle file contain this text'
			print
			continue 
			
		else:
			for i in range(len(pickles)):
				j = (index + i) % len(pickles)
				if key in pickles[j]:
					index = j
					specific = True
					break
			else:
				print key, 'not found'
				continue
		
		index = index % len(pickles)
		
		display(pickles, index, specific)

		

	
except KeyboardInterrupt:
	None






