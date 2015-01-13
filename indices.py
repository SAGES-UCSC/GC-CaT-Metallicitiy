#! /usr/bin/env python

import numpy as np

#assume constant linear dispersion
#C01 = Cenarro et al 2001 MNRAS 326 959

# calculates a sum of a band allowing for fractional pixels
def bandSum(band, wavelengths, fluxes, edges=True):

	dwavelength = wavelengths[1] - wavelengths[0]
	inband = (wavelengths > band[0] - dwavelength / 2) & (wavelengths < band[1] + dwavelength / 2)
	inwavelengths = wavelengths[inband]
	influxes = fluxes[inband]

	total = influxes[1:-1].sum()
	if edges:
		total += influxes[0] * (inwavelengths[:2].mean() - band[0]) / dwavelength
		total += influxes[-1] * (band[1] - inwavelengths[-2:].mean()) / dwavelength

	return total

class Index:

	def __init__(self, name, mainpassbands, continuumpassbands, mainweights=None, atomic=True, flux=False):

		self.name = name
		self.mainpassbands = np.array(mainpassbands)
		self.continuumpassbands = np.array(continuumpassbands)
		if mainweights is None:
			self.mainpassweights = np.ones(self.mainpassbands.shape[0])
		else:
			self.mainpassweights = np.array(mainweights)
		
			
		self.atomic = atomic
		self.flux = flux

	def __str__(self):
		return self.name


	def __call__(self, wavelengths, fluxes, sigmas=None, normalized=False, calcerrors=False):
		return self.calculateIndex(wavelengths, fluxes, sigmas=sigmas, normalized=normalized, calcerrors=calcerrors)

	def calculateIndex(self, wavelengths, fluxes, sigmas=None, normalized=False, calcerrors=False):
		
		
		if sigmas == None:
			variances = fluxes
		else:
			variances = sigmas**2

		if fluxes.size != wavelengths.size or fluxes.size != variances.size:

			raise Exception('Input array size mismatch')	


		dwavelength = wavelengths[1] - wavelengths[0]

		#
		if not normalized:

			if wavelengths[0] > min(self.mainpassbands.min(), self.continuumpassbands.min()) or wavelengths[-1] < max(self.mainpassbands.max(), self.continuumpassbands.max()):
				raise Exception('Index not in wavelength range')
	
			#correspond to eqs a15 to a19 of C01
			Sigma1 = 0
			Sigma2 = 0
			Sigma3 = 0
			Sigma4 = 0
			Sigma5 = 0

			inversevariances =  1 / variances 
			sigma1seq = inversevariances
			sigma2seq = wavelengths * inversevariances
			sigma3seq = wavelengths**2 * inversevariances
			sigma4seq = fluxes * inversevariances
			sigma5seq = wavelengths * fluxes * inversevariances
			

			for passband in self.continuumpassbands:
				inpassband = (wavelengths > passband[0]) & (wavelengths < passband[1])
				Sigma1 += sigma1seq[inpassband].sum()
				Sigma2 += sigma2seq[inpassband].sum()
				Sigma3 += sigma3seq[inpassband].sum()
				Sigma4 += sigma4seq[inpassband].sum()
				Sigma5 += sigma5seq[inpassband].sum()

			# eq a14 of C01
			Lambda = Sigma1 * Sigma3 - Sigma2 **2

			#eqs a12 and a13 of C01
			alpha1 = (Sigma3 * Sigma4 - Sigma2 * Sigma5) / Lambda
			alpha2 = (Sigma1 * Sigma5 - Sigma2 * Sigma4) / Lambda

			continuum = alpha1 + alpha2 * wavelengths

			#Note that this has not been properly checked
			if calcerrors:
				
				error = 0
				fluxerror = 0
				continuumvariance = 0
				
				#eq a25
				continuummatrix = np.outer(np.ones(wavelengths.size), (sigma1seq * Sigma3 -  sigma2seq * Sigma2) / Lambda) + np.outer(wavelengths, (sigma2seq * Sigma1 - sigma1seq * Sigma2) / Lambda)
				
				#eq a26
				for passband in self.continuumpassbands:
					inpassband = (wavelengths > passband[0]) & (wavelengths < passband[1])
					continuumvariance += np.dot(continuummatrix[:,inpassband], variances[inpassband]) 
				
				#eqs a28 to a30
				a11 = (Sigma1 * Sigma3 * Sigma3 - Sigma2 * Sigma2 * Sigma3) / Lambda**2
				a12 = (Sigma2 * Sigma2 * Sigma2 - Sigma1 * Sigma2 * Sigma3) / Lambda**2
				a22 = (Sigma1 * Sigma1 * Sigma3 - Sigma1 * Sigma2 * Sigma2) / Lambda**2
				
			
				
				if self.flux:
					#flux excess
					for i in range(self.mainpassweights.shape[0]):
						outerpassband = self.mainpassbands[i]
						weight = self.mainpassweights[i]
						inouterpassband = (wavelengths > outerpassband[0]) & (wavelengths < outerpassband[1])
						
						fluxerror += weight**2 * (variances[inouterpassband] + continuumvariance[inouterpassband]).sum()
						error += weight**2 * ((continuum[inouterpassband]**2 * variances[inouterpassband] + fluxes[inouterpassband]**2 * continuumvariance[inouterpassband]) / continuum[inouterpassband]**4).sum()
						
						for j in inouterpassband.nonzero()[0]:
							for k in range(self.mainpassbands.shape[0]):
								innerpassband = self.mainpassbands[k]
								ininnerpassband = (wavelengths > innerpassband[0]) & (wavelengths < innerpassband[1])
								for l in ininnerpassband.nonzero()[0]:
									if (i != k) or (j != l):
										fluxerror += self.mainpassweights[i] * self.mainpassweights[k] * (a11 + a12 * (wavelengths[j] + wavelengths[l]) + a22 * wavelengths[j] * wavelengths[l])
										error += self.mainpassweights[i] * self.mainpassweights[k] * fluxes[j] * fluxes[l] * (a11 + a12 * (wavelengths[j] + wavelengths[l]) + a22 * wavelengths[j] * wavelengths[l]) / (continuum[j]**2 * continuum[l]**2 )
					
					fluxerror = fluxerror**.5 * dwavelength
				else:	
					#eq a23
					for i in range(self.mainpassweights.shape[0]):
						
						outerpassband = self.mainpassbands[i]
						weight = self.mainpassweights[i]
						inouterpassband = (wavelengths > outerpassband[0]) & (wavelengths < outerpassband[1])
						
						error += weight**2 * ((continuum[inouterpassband]**2 * variances[inouterpassband] + fluxes[inouterpassband]**2 * continuumvariance[inouterpassband]) / continuum[inouterpassband]**4).sum()
						
						for j in inouterpassband.nonzero()[0]:
							for k in range(self.mainpassbands.shape[0]):
								innerpassband = self.mainpassbands[k]
								ininnerpassband = (wavelengths > innerpassband[0]) & (wavelengths < innerpassband[1])
								for l in ininnerpassband.nonzero()[0]:
									if (i != k) or (j != l):
										error += self.mainpassweights[i] * self.mainpassweights[k] * fluxes[j] * fluxes[l] * (a11 + a12 * (wavelengths[j] + wavelengths[l]) + a22 * wavelengths[j] * wavelengths[l]) / (continuum[j]**2 * continuum[l]**2 )
	
				error = error**.5 * dwavelength					

				
			else:
				error = 0
				fluxerror = 0

		# assume the spectra is normalized
		else:
			if wavelengths[0] > self.mainpassbands.min() or wavelengths[-1] < self.mainpassbands.max():
				print wavelengths[0], self.mainpassbands.min(), wavelengths[-1], self.mainpassbands.max()
				raise Exception('Index not in wavelength range')
			continuum = np.zeros(fluxes.size) + 1
			error = 0

		

		total = 0

		normdepths = 1 - fluxes / continuum 

		# correspond to eq a9 of C01
		for i in range(self.mainpassbands.shape[0]):
			total += dwavelength * self.mainpassweights[i] * bandSum(self.mainpassbands[i], wavelengths, normdepths)
		
		if False:# self.name == 'All Metals':
			import matplotlib.pyplot as plt
			plt.figure()
			plt.plot(wavelengths, fluxes)
			plt.plot(wavelengths, continuum)
			plt.vlines(self.continuumpassbands.flatten(), continuum.mean(), 1.1 * continuum.mean())
			plt.vlines(self.mainpassbands.flatten(), 0.9 * continuum.mean(), continuum.mean())
			plt.title(self.name)

		if not self.flux:		
			return total, error
		else:
			fluxtotal = 0
			excesses = fluxes - continuum
			for i in range(self.mainpassbands.shape[0]):
				fluxtotal += dwavelength * self.mainpassweights[i] * bandSum(self.mainpassbands[i], wavelengths, excesses)
				
			
			
			return total, error, fluxtotal, fluxerror 


class Ratio:
	
	def __init__(self, name, blueband, redband):
		
		self.name = name
		self.blueband = blueband
		self.redband = redband
		
	
	def __call__(self, wavelengths, fluxes, sigmas=None):
		return self.calculateIndex(wavelengths, fluxes, sigmas=sigmas)

	def calculateIndex(self, wavelengths, fluxes, sigmas=None):
		
		if sigmas == None:
			sigmas = np.zeros(fluxes.size)
			
		
		if wavelengths[0] > self.blueband[0] or wavelengths[-1] < self.redband[-1]:
			raise Exception('Index not in wavelength range')


		varis = sigmas**2


		edges = (wavelengths[:-1] + wavelengths[1:]) / 2
		
		lower_edges = np.hstack((3 * wavelengths[0] / 2 - wavelengths[1] / 2, edges))
		upper_edges = np.hstack((edges, 3 * wavelengths[-1] / 2 - wavelengths[-2] / 2))
			
			
		blue_flux = 0
		blue_vari = 0
		
		red_flux = 0
		red_vari = 0	
		
		for i in range(wavelengths.size):
			
			
			if upper_edges[i] >= self.blueband[0] and lower_edges[i] <= self.blueband[1]:
				if lower_edges[i] <= self.blueband[0]:
					
					factor = (upper_edges[i] - self.blueband[0]) / (upper_edges[i] - lower_edges[i])
					
				elif upper_edges[i] >= self.blueband[0]:
					
					factor = (self.blueband[1] - lower_edges[i]) / (upper_edges[i] - lower_edges[i])
				
				else:
					factor = 1
					
				blue_flux += fluxes[i] * factor
				blue_vari += varis[i] * factor**2
				
			if upper_edges[i] >= self.redband[0] and lower_edges[i] <= self.redband[1]:
				
				if lower_edges[i] <= self.redband[0]:
					
					factor = (upper_edges[i] - self.redband[0]) / (upper_edges[i] - lower_edges[i])
					
				elif upper_edges[i] >= self.redband[0]:
					
					factor = (self.redband[1] - lower_edges[i]) / (upper_edges[i] - lower_edges[i])
				
				else:
					factor = 1
					
				red_flux += fluxes[i] * factor
				red_vari += varis[i] * factor**2			
		

#		print blue_flux, blue_vari**0.5
#		print red_flux, red_vari**0.5

		ratio = blue_flux / red_flux
		
		error = (blue_vari / red_flux**2 + blue_flux**2 / red_flux**4 * red_vari)**0.5
		
		return ratio, error
	
	
TiO_89 = Ratio('TiO 8855', np.array([8835.0, 8855.0]), np.array([8870.0, 8890.0]))

# Only meant for normalized spectra
CaT_gc = Index('CaT gc', np.array([[8490.0, 8506.0], [8532.0, 8552.0], [8653.0, 8671.0]]), None, np.array([1, 1, 1]))
Pa12_gc = Index('Pa 12 gc', np.array([[8730.0, 8772.0]]), None, np.array([1]))

Fe8387 = Index('Fe 8387', np.array([[]]), None, np.array([1]))
Ti8435 = Index('Ti 8435', np.array([[8432.0, 8438.0]]), None, np.array([1]))
Fe8514 = Index('Fe 8514', np.array([[8511.0, 8517.0]]), None, np.array([1]))
Si8556 = Index('Fe 8556', np.array([[8553.0, 8559.0]]), None, np.array([1]))
Fe8582 = Index('Fe 8582', np.array([[8579.0, 8585.0]]), None, np.array([1]))
Fe8611 = Index('Fe 8611', np.array([[8608.5, 8614.5]]), None, np.array([1]))
Fe8621 = Index('Fe 8621', np.array([[8608.0, 8614.0]]), None, np.array([1]))
Si8648 = Index('Si 8648', np.array([[8645.0, 8651.0]]), None, np.array([1]))
Fe8674 = Index('Fe 8674', np.array([[8671.5, 8677.5]]), None, np.array([1]))
Fe8688 = Index('Fe 8688', np.array([[8685.0, 8691.0]]), None, np.array([1]))

Fe8824 = Index('Fe 8824', np.array([[8821.0, 8827.0]]), None, np.array([1]))

Pa17 = Index('Pa 17', np.array([[8461.0, 8474.0]]), None, np.array([1]))
Pa14 = Index('Pa 16', np.array([[8592.0, 8604.0]]), None, np.array([1]))

AllMetals = Index('All Metals', np.array([[8375.5, 8392.0],
										[8410.4, 8414.0],
										[8424.5, 8428.0],
										[8432.5, 8440.9],
										[8463.7, 8473.0],
										[8512.8, 8519.0],
										[8580.8, 8583.5],
										[8595.7, 8601.0],
										[8609.0, 8613.5],
										[8620.2, 8623.3],
										[8673.2, 8676.5],
										[8686.8, 8690.7],
										[8820.5, 8827.0],
										[8836.0, 8840.5]]), 
				np.array([[8392.0, 8393.5],
						[8399.4, 8400.9],
						[8402.7, 8410.3],
						[8414.5, 8422.1],
						[8428.6, 8432.3],
						[8441.4, 8445.2],
						[8447.9, 8449.4],
						[8451.5, 8455.4],
						[8458.0, 8463.0],
						[8474.0, 8493.3],
						[8505.3, 8512.1],
						[8519.2, 8525.2],
						[8528.3, 8531.3],
						[8552.3, 8554.9],
						[8557.5, 8580.4],
						[8583.9, 8595.3],
						[8601.2, 8608.4],
						[8613.9, 8619.4],
						[8624.3, 8646.6],
						[8649.8, 8652.5],
						[8676.9, 8678.1],
						[8684.0, 8686.1],
						[8692.7, 8697.6],
						[8700.3, 8708.9],
						[8714.5, 8726.8],
						[8731.5, 8733.2],
						[8737.6, 8740.8],
						[8743.3, 8746.1],
						[8754.5, 8755.4],
						[8759.0, 8762.2],
						[8768.0, 8771.5],
						[8775.5, 8788.7],
						[8797.6, 8802.2],
						[8811.0, 8820.0],
						[8828.0, 8835.0]]))



WeakMetals = Index('All Metals', np.array([[8375.5, 8392.0],
										[8393.5, 8398.5],
										[8410.4, 8414.0],
										[8424.5, 8428.0],
										[8432.5, 8440.9],
										[8463.7, 8473.0],
										[8512.8, 8519.0],
										[8555.0, 8557.4],
										[8580.8, 8583.5],
										[8595.7, 8601.0],
										[8609.0, 8613.5],
										[8620.2, 8623.3],
										[8647.5, 8649.7],
										[8673.2, 8676.5],
										[8686.8, 8690.7],
										[8709.0, 8714.0],
										[8733.9, 8737.3],
										[8746.6, 8753.8],
										[8755.7, 8758.7],
										[8762.5, 8767.5],
										[8771.9, 8775.0],
										[8789.3, 8794.7],
										[8820.5, 8827.0],
										[8836.0, 8840.5]]), 
				np.array([[8392.0, 8393.5],
						[8399.4, 8400.9],
						[8402.7, 8410.3],
						[8414.5, 8422.1],
						[8428.6, 8432.3],
						[8441.4, 8445.2],
						[8447.9, 8449.4],
						[8451.5, 8455.4],
						[8458.0, 8463.0],
						[8474.0, 8493.3],
						[8505.3, 8512.1],
						[8519.2, 8525.2],
						[8528.3, 8531.3],
						[8552.3, 8554.9],
						[8557.5, 8580.4],
						[8583.9, 8595.3],
						[8601.2, 8608.4],
						[8613.9, 8619.4],
						[8624.3, 8646.6],
						[8649.8, 8652.5],
						[8676.9, 8678.1],
						[8684.0, 8686.1],
						[8692.7, 8697.6],
						[8700.3, 8708.9],
						[8714.5, 8726.8],
						[8731.5, 8733.2],
						[8737.6, 8740.8],
						[8743.3, 8746.1],
						[8754.5, 8755.4],
						[8759.0, 8762.2],
						[8768.0, 8771.5],
						[8775.5, 8788.7],
						[8797.6, 8802.2],
						[8811.0, 8820.0],
						[8828.0, 8835.0]]))

CaT = Index('CaT', np.array([[8490.0, 8506.0], [8532.0, 8552.0], [8653.0, 8671.0]]), 
				np.array([[ 8399.37361512, 8400.87320799], [ 8402.67271944, 8411.07043951], [ 8414.06962524, 8416.76889239], [ 8418.8683224,  8422.16742669], [ 8427.86587954, 8433.2644138 ], [ 8441.36221517, 8445.26115656], [ 8447.96042367, 8449.4600165 ], [ 8450.95960934, 8456.05822497], [ 8457.5578178,  8464.45594481], [ 8472.55374605, 8480.65154725], [ 8482.75097719, 8494.14788252], [ 8505.24486923, 8512.44291463], [ 8518.74120433, 8525.33941257], [ 8528.33859812, 8531.33778367], [ 8552.03216386, 8555.03134937], [ 8557.43069778, 8581.12426318], [ 8583.82353011, 8596.1201905 ], [ 8600.01913159, 8610.21636209], [ 8613.21554752, 8615.91481441], [ 8616.81457003, 8620.41359254], [ 8623.11285941, 8647.40626113], [ 8649.80560944, 8652.50487627], [ 8676.79827766, 8678.59778886], [ 8683.99632246, 8686.39567072], [ 8690.8944487,  8698.09249345], [ 8700.4918417,  8709.18947907], [ 8714.28809406, 8727.78442781], [ 8729.58393897, 8734.38263538], [ 8737.08190212, 8741.28076147], [ 8743.08027262, 8746.07945786], [ 8753.87733948, 8755.67685062], [ 8758.67603584, 8762.57497663], [ 8767.37367298, 8772.17236931], [ 8774.871636,   8789.26772493], [8795.2, 8802.2], [8811, 8820], [8828, 8835]]))

NaD = Index('NaD', np.array([[8180.0, 8200.0]]), np.array([[8164.0, 8173.0], [8233.0, 8244.0]])) # 2012MNRAS.424..157V

Ha = Index('Ha', np.array([[6552.8, 6572.9]]), np.array([[6520.6, 6543.0], [6582.0, 6590.0], [6596.0, 6604.2], [6610.5, 6640.4], [6645.7, 6660.5], [6665.5, 6675.4], [6681.7, 6714.0]]))


CaT_AZ_1 = Index('CaT A&Z 1', np.array([[8490.0, 8506.0]]), np.array([[8474.0, 8489.0], [8521.0, 8531.0]]), np.array([1]))
CaT_AZ_2 = Index('CaT A&Z 2', np.array([[8532.0, 8552.0]]), np.array([[8521.0, 8531.0], [8555.0, 8595.0]]), np.array([1]))
CaT_AZ_3 = Index('CaT A&Z 3', np.array([[8653.0, 8671.0]]), np.array([[8626.0, 8650.0], [8695.0, 8725.0]]), np.array([1]))

CaT_C01 = Index('CaT C01', np.array([[8484.0, 8513.0], [8522.0, 8562.0], [8642.0, 8682.0]]), np.array([[8474.0, 8484.0], [8563.0, 8577.0], [8619.0, 8642.0], [8700.0, 8725.0], [8776.0, 8792.0]]), np.array([1, 1, 1]))
PaT_C01 = Index('PaT C01', np.array([[8461.0, 8474.0], [8577.0, 8619.0], [8730.0, 8772.0]]), np.array([[8474.0, 8484.0], [8563.0, 8577.0], [8619.0, 8642.0], [8700.0, 8725.0], [8776.0, 8792.0]]), np.array([1, 1, 1]))
CaTS_C01 = Index('CaT* C01', np.array([[8484.0, 8513.0], [8522.0, 8562.0], [8642.0, 8682.0], [8461.0, 8474.0], [8577.0, 8619.0], [8730.0, 8772.0]]), np.array([[8474.0, 8484.0], [8563.0, 8577.0], [8619.0, 8642.0], [8700.0, 8725.0], [8776.0, 8792.0]]), np.array([1, 1, 1, -.93, -.93, -.93]))

MgI = Index('MgI', np.array([[8802.5, 8811.0]]), np.array([[8781.0, 8787.0], [8831.0, 8835.5]]), np.array([1]))

HaA = Index('HaA', np.array([[6554.0, 6575.0]]), np.array([[6515.0, 6540.0], [6575.0, 6585.0]]), np.array([1]))
HaF = Index('HaF', np.array([[6554.0, 6568.0]]), np.array([[6515.0, 6540.0], [6568.0, 6575.0]]), np.array([1]))
HaW = Index('HaW', np.array([[6550.0, 6575.0]]), np.array([[6600.0, 6800.0]]), np.array([1]))
Sky = Index('Sky', np.array([[8605.0, 8695.5]]), np.array([[8478.0, 8489.0], [8813.0, 8822.0]]), np.array([1 / 90.5]), flux=True)
	
CaT_gal = Index('CaT gal', np.array([[8483.0, 8513.0], [8527.0, 8557.0], [8647.0, 8677.0]]), np.array([[8474.0, 8483.0], [8514.0, 8526.0], [8563.0, 8577.0], [8619.0, 8684.0], [8680.0, 8705.0]]), np.array([0.4, 1.0, 1.0]))