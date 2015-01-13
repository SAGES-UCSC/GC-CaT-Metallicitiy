#! /usr/bin/env python

import numpy as np
import pyfits
from utils import uopen


### This is a group of functions meant to manipulate 1d spectra as numpy arrays

# returns the dispersion and flux as a tuple of arrays from a fits file. The WCS is used to calculate the dispersion. Only simple linear or log dispersion is handled
def getspectrum(name, axis=None):

	fits = pyfits.open(uopen(name, 'rb'))
	
	data = fits[0].data
	
	if len(data.shape) > 1:
		
		data = data[axis]
			
	
	
	crval = fits[0].header['CRVAL1']
	cdelt = fits[0].header['CDELT1']
	crpix = fits[0].header['CRPIX1']
	wavelengths = np.arange(data.size)
	wavelengths =  (wavelengths - crpix + 1) * cdelt + crval
	if 'DC-FLAG' in fits[0].header and fits[0].header['DC-FLAG']:
		#log dispersion
		wavelengths = np.exp(wavelengths * np.log(10))
	fits.close()

	return (wavelengths, data)

def load(name, axis=None):
	
	return getspectrum(name, axis)

def writespectrum(name, wavelengths, fluxes):

	hdu = pyfits.PrimaryHDU(fluxes)
	hdu.header.update('CRPIX1', 1)
	hdu.header.update('CRVAL1', wavelengths[0])
	hdu.header.update('CDELT1', wavelengths[1] - wavelengths[0])
	hdu.header.update('DC-FLAG', 0)
	hdulist = pyfits.HDUList([hdu])
	hdulist.writeto(name)
	hdulist.close()


def save(name, wavelengths, fluxes):
	
	writespectrum(name, wavelengths, fluxes)

#boxcar smooth an array using of a kernel of 2 * 'kernel' + 1 pixels
def smooth(data, kernel=3):
	output = np.empty(data.size)

	for i in range(data.size):
		if i - kernel < 0:
			output[i] = data[0:i + kernel + 1].mean() 
		elif data.size - (i + kernel + 1) < 0:
			output[i] = data[i - kernel:data.size].mean()
		else:
			output[i] = data[i - kernel:i + kernel + 1].mean()
	return output

def updateheader(name, pairs):
	fits = pyfits.open(name, mode='update')
	header = fits[0].header
	for key in pairs.keys():
		header.update(key, pairs[key])
	fits.close()

def listheader(name, keys):
	fits = pyfits.open(uopen(name, 'rb'))
	header = fits[0].header
	fits.close()
	if isinstance(keys, str):
		return header[keys]
	else:
		output = []
		
		for key in keys:
			output.append(header[key])
		return output
