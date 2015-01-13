#! /usr/bin/env python

import numpy as np

def normalise(wavelengths, fluxes, order, mask=None, iters=None, lowsigma=0.005, upsigma=0.010, ivars=None):
	return fluxes / getnormalisation(wavelengths, fluxes, order, mask, iters, lowsigma, upsigma, ivars)

def getnormalisation(wavelengths, fluxes, order, maskregions=None, iters=None, lowsigma=0.005, upsigma=0.010, ivars=None):

	if wavelengths.size != fluxes.size:
		raise Exception('Wavelength and flux arrays are different size')
	
	if ivars is None or ivars.size != wavelengths.size:
		ivars = np.ones(wavelengths.size)
	
	regionmask = np.ones(wavelengths.size, dtype=np.bool_)
	if maskregions is not None:
		for maskregion in maskregions:
			regionmask = regionmask & ~((wavelengths > maskregion[0]) & (wavelengths < maskregion[1]))

	polydomain = np.linspace(-1, 1, wavelengths.size) #Domain of the fitted polynomial

	polys = [np.poly1d([1]), np.poly1d([1, 0])]

	while len(polys) <= order:

		newpoly = np.poly1d([2, 0]) * polys[-1] - polys[-2]
		polys.append(newpoly)

	polys = polys[-1::-1]

	basisvectors = np.zeros((wavelengths.size, order + 1))
	weightedbasisvectors = basisvectors.copy()
	
	for i in range(basisvectors.shape[1]):
		basisvectors[:, i] = polys[i](polydomain)
		weightedbasisvectors[:, i] = polys[i](polydomain) * ivars


	#sortedfluxes = fluxes[fluxes.argsort()]

	#print sortedfluxes[:10], sortedfluxes[-10:-1]
	#lower10 = sortedfluxes[int(sortedfluxes.size / 10)]

	#lower10mask = fluxes >= lower10
	#print lower10mask


	pixelmask = regionmask
	masksize = fluxes[pixelmask].size

	beta = np.linalg.lstsq(weightedbasisvectors[pixelmask,:], fluxes[pixelmask] * ivars[pixelmask])[0]
	model = np.dot(basisvectors, beta)
	#print masksize

	if iters:
		while True:

			oneprecut = ((fluxes - model) / model > -lowsigma) & ((fluxes - model) / model < upsigma)

			pixelmask = oneprecut#& regionmask

			maskedfluxes = fluxes[pixelmask] 
			beta = np.linalg.lstsq(weightedbasisvectors[pixelmask,:], maskedfluxes * ivars[pixelmask])[0]
			model = np.dot(basisvectors, beta)
			
			if masksize == maskedfluxes.size:
				#print masksize
				break
			else:
				masksize = maskedfluxes.size
				#print masksize
	

	return model


gccatmask = np.array([[8488.0, 8508.0], #Ca1
[8530.1, 8554.1], #Ca2
[8652.1, 8672.1], #Ca3
[8803.8, 8809.8], #MgI
[8432.0, 8438.0], #TiI
[8511.1, 8517.1], #FeI
[8671.8, 8677.8], #FeI
[8685.6, 8691.6], #FeI
[8821.4, 8827.4]]) #FeI

pamask = np.array([[8457.3, 8477.3],
[8492.5, 8512.5],
[8535.4, 8555.4],
[8578.4, 8618.4],
[8655.0, 8675.0],
[8730.0, 8770.0]])


mask = np.vstack((gccatmask, pamask))


newmask = np.array([[8432.0, 8438.0],
[8462.5, 8474.0],
[8491.0, 8506.4],
[8510.0, 8520.0],
[8524.0, 8552.5],
[8553.8, 8558.0],
[8579.7, 8585.5],
[8594.7, 8600.9],
[8609.3, 8615.0],
[8619.6, 8624.0],
[8646.5, 8677.0],
[8685.5, 8693.6],
[8708.2, 8715.0],
[8733.0, 8738.0],
[8740.0, 8743.5],
[8745.6, 8753.5],
[8755.6, 8758.7],
[8762.2, 8768.0],
[8769.9, 8775.8],
[8777.5, 8781.0],
[8783.3, 8785.8],
[8788.1, 8795.3],
[8800.0, 8810.5],
[8817.7, 8828.0],
[8832.3, 8842.2]])


ssp_mask = np.array([[8340.0, 8366.50],
[8373.00, 8402.67271944],
[8411.07043951, 8414.0696252],
[8416.76889239, 8418.8683224],
[8422.16742669, 8427.86587954],
[8433.2644138, 8441.36221517],
[8445.26115656, 8447.96042367],
[8449.4600165, 8450.95960934],
[8456.05822497, 8457.5578178], 
[8464.45594481, 8472.55374605],
[8480.65154725, 8482.75097719],
[8494.14788252, 8505.24486923],
[8512.44291463, 8518.74120433],
[8525.33941257, 8528.33859812],
[8531.33778367, 8552.03216386],
[8555.03134937, 8557.43069778],
[8581.12426318, 8583.82353011],
[8596.1201905, 8600.01913159],
[8610.21636209, 8613.21554752],
[8620.41359254, 8623.11285941],
[8647.40626113, 8649.80560944],
[8652.50487627, 8676.79827766],
[8678.59778886, 8683.99632246],
[8686.39567072, 8690.8944487],
[8698.09249345, 8700.4918417],
[8709.18947907, 8714.28809406],
[8727.78442781, 8729.58393897],
[8734.38263538, 8737.08190212],
[8741.28076147, 8743.08027262],
[8746.07945786, 8753.87733948],
[8755.67685062, 8758.67603584],
[8762.57497663, 8767.37367298],
[8772.17236931, 8774.871636],
[8789.26772493, 8794.470055],
[8802.966155, 8808.913425],
[8818.259135, 8826.755235],  
[8836.950555, 8840.]])

Halpha_mask = np.array([[6513.8, 6520.6],
					[6543.0, 6582.0],
					[6590.0, 6596.0],
					[6604.2, 6610.5],
					[6640.4, 6645.7],
					[6660.5, 6665.5],
					[6675.4, 6681.7],
					[6714.0, 6720.0]])


NaD_mask = np.array([[8018.8, 8031.5],
					[8040.7, 8051.5],
					[8087.5, 8100.1],
					[8157.0, 8164.0],
					[8177.0, 8224.9],
					[8249.5, 8259.9],
					[8300.1, 8314.0],
					[8323.2, 8350.8],
					[8361.1, 8367.0],
					[8374.6, 8390.7]])

red_mask = np.array([[8018.8, 8031.5],
					[8040.7, 8051.5],
					[8087.5, 8100.1],
					[8157.0, 8164.0],
					[8177.0, 8224.9],
					[8249.5, 8259.9],
					[8300.1, 8314.0],
					[8323.2, 8350.8],
					[8361.1, 8367.0],
					[8374.6, 8390.7],
					[8340.0, 8366.5],
					[8373.00, 8402.67271944],
					[8411.07043951, 8414.0696252],
					[8416.76889239, 8418.8683224],
					[8422.16742669, 8427.86587954],
					[8433.2644138, 8441.36221517],
					[8445.26115656, 8447.96042367],
					[8449.4600165, 8450.95960934],
					[8456.05822497, 8457.5578178], 
					[8464.45594481, 8472.55374605],
					[8480.65154725, 8482.75097719],
					[8494.14788252, 8505.24486923],
					[8512.44291463, 8518.74120433],
					[8525.33941257, 8528.33859812],
					[8531.33778367, 8552.03216386],
					[8555.03134937, 8557.43069778],
					[8581.12426318, 8583.82353011],
					[8596.1201905, 8600.01913159],
					[8610.21636209, 8613.21554752],
					[8620.41359254, 8623.11285941],
					[8647.40626113, 8649.80560944],
					[8652.50487627, 8676.79827766],
					[8678.59778886, 8683.99632246],
					[8686.39567072, 8690.8944487],
					[8698.09249345, 8700.4918417],
					[8709.18947907, 8714.28809406],
					[8727.78442781, 8729.58393897],
					[8734.38263538, 8737.08190212],
					[8741.28076147, 8743.08027262],
					[8746.07945786, 8753.87733948],
					[8755.67685062, 8758.67603584],
					[8762.57497663, 8767.37367298],
					[8772.17236931, 8774.871636],
					[8789.26772493, 8794.470055],
					[8802.966155, 8808.913425],
					[8818.259135, 8826.755235],  
					[8836.950555, 8840.0],
					[8852.0, 8872.0]])

blue_mask = np.array([[6499.0, 6502.0],
					[6513.8, 6520.6],
					[6543.0, 6582.0],
					[6590.0, 6596.0],
					[6604.2, 6610.5],
					[6640.4, 6645.7],
					[6660.5, 6665.5],
					[6675.4, 6681.7],
					[6714.0, 6720.0],
					[6746.2, 6754.4],
					[6763.1, 6775.8],
					[6833.3, 6845.1],
					[6907.5, 6921.1],
					[6941.1, 6947.6],
					[6972.3, 6983.0],
					[7012.4, 7100.0]])

# check that everything works
def test():

	import matplotlib.pyplot as plt
	import pickle
	
#	datum = pickle.load(open('test.z-0.38t17.78.rv0000.s2n100.n000.pickle'))
#	
#	wavelengths = datum.fitwavelengths
#	fluxes = datum.fitfluxes
#	
#	
#	plt.figure()
#	plt.hlines(np.ones(mask.shape[0]) * fluxes.mean() * .95, mask[:,0], mask[:,1], 'r')
#	plt.hlines(np.ones(newmask.shape[0]) * fluxes.mean() * .9, newmask[:,0], newmask[:,1], 'b')	
#	plt.plot(wavelengths, fluxes, 'k-')
##	plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 7, mask, False, 0, 0), 'b:')
##
##	plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 7, mask, True, .005, .005), 'r:')
##	plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 7, mask, True, .003, .005), 'y:')
##	plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 7, mask, True, .002, .005), 'g:')
##	plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 7, mask, True, .005, .008), 'k:')
##
##	plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 7, newmask, False, 0, 0), 'b--')
##
##	plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 7, newmask, True, .005, .005), 'r--')
##	plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 7, newmask, True, .003, .005), 'y--')
##	plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 7, newmask, True, .002, .005), 'g--')
##	plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 7, newmask, True, .005, .008), 'k--')
#
#	#plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 9, newmask, True, .005, .015), ls=':', c='#000000')
##	plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 9, newmask, True, .004, .015), ls=':', c='#004000')
#	#plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 9, newmask, True, .0035, .015), ls=':', c='#006000')
#	#plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 9, newmask, True, .003, .015), ls=':', c='#008000')
#	#plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 9, newmask, True, .002, .015), ls=':', c='#00c000')
#	
##	plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 9, newmask, True, .005, .010), ls=':', c='#400000')
##	plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 9, newmask, True, .004, .010), ls=':', c='r')
##	plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 9, newmask, True, .003, .010), ls=':', c='#408000')
##	plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 9, newmask, True, .002, .010), ls=':', c='#40c000')	
##
##	plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 9, newmask, True, .005, .005), ls=':', c='#800000')
##	plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 9, newmask, True, .004, .005), ls=':', c='#804000')
##	plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 9, newmask, True, .003, .005), ls=':', c='#808000')
##	plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 9, newmask, True, .002, .005), ls=':', c='#80c000')
#	plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 8, newmask, True, .004, .010), ls=':', c='y')
##	plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 7, newmask, True, .004, .010), ls=':', c='g')



	import glob
	
	fits = glob.glob('explore_test.z-0.38t10.00.rv0000.s2n999.n000.pickle')
	
	for fit in fits:
		datum = pickle.load(open(fit))
		
		wavelengths = datum.fitwavelengths
		fluxes = datum.fitfluxes
		
		
		plt.figure()
		plt.hlines(np.ones(mask.shape[0]) * fluxes.mean() * .95, mask[:,0], mask[:,1], 'r')
		plt.hlines(np.ones(newmask.shape[0]) * fluxes.mean() * .9, newmask[:,0], newmask[:,1], 'b')	
		plt.plot(wavelengths, fluxes, 'k-')		
		plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 8, newmask, True, .004, .010), ls=':', c='g')
		#plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 8, newmask, False, .004, .010), ls=':', c='y')
		#plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 8, mask, True, .004, .010), ls=':', c='c')
		plt.title(fit)

		if fit == 'explore_test.z-0.38t10.00.rv0000.s2n999.n000.pickle':
			continuum = normalise(wavelengths, fluxes, 8, newmask, True, .004, .010) > .99
			plt.plot(wavelengths[continuum], fluxes.mean() + 0 * fluxes[continuum], 'm,')
		
			dispersion = wavelengths[1] - wavelengths[0]
		
			starts = []
			ends = []
			
			
			inband = False	
			
			for i in range(wavelengths.size):
				if not inband and not continuum[i]:
					inband = True
					starts.append(wavelengths[i] - dispersion / 2)
					
				if inband and continuum[i]:
					inband = False
					ends.append(wavelengths[i] - dispersion / 2)
					
			
			if len(starts) > len(ends):
				ends.append(wavelengths[-1] + dispersion / 2)
				
			print starts
			print ends
			
			bands = []
			for i in range(len(starts)):
				bands.append([starts[i], ends[i]])
				
			bands = np.array(bands)
				
			plt.hlines(np.ones(bands.shape[0]) * fluxes.mean() * 1.01, bands[:,0], bands[:,1], 'm')
			plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 8, bands, True, .004, .010), ls=':', c='c')
			plt.plot(wavelengths, getnormalisation(wavelengths, fluxes, 8, bands, False, .004, .010), ls=':', c='y')
			print bands
		
	
	plt.show()

if __name__ == "__main__":
	test()
