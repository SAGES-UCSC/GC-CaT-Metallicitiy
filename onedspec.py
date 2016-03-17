#! /usr/bin/env python

import numpy as np
import astropy.io.fits as fits
import os

### This is a group of functions meant to manipulate 1d spectra as numpy arrays

# returns the dispersion and flux as a tuple of arrays from a fits file. The WCS is used to calculate the dispersion. Only simple linear or log dispersion is handled
def load(name, extension=0, axis=0):

	hdulist = fits.open(open(os.path.abspath(os.path.expanduser(name)), 'rb'))
	data = hdulist[extension].data
	
	if len(data.shape) > 1:
		data = data[axis]
	
	crval = hdulist[extension].header['CRVAL1']
	cdelt = hdulist[extension].header['CDELT1']
	
	if 'CRPIX1' in hdulist[extension].header:
		crpix = hdulist[extension].header['CRPIX1']
	else:
		crpix = 1
	wavelengths = np.arange(data.size)
	wavelengths =  (wavelengths - crpix + 1) * cdelt + crval
	if 'DC-FLAG' in hdulist[extension].header and hdulist[extension].header['DC-FLAG']:
		#log dispersion
		wavelengths = np.exp(wavelengths * np.log(10))
	hdulist.close()

	return (wavelengths, data)

def load_with_errors(name):

	hdulist = fits.open(open(os.path.abspath(os.path.expanduser(name)), 'rb'))
	fluxes = hdulist[0].data
	errors = hdulist[1].data
	

	crval = hdulist[0].header['CRVAL1']
	cdelt = hdulist[0].header['CDELT1']
	
	if 'CRPIX1' in hdulist[0].header:
		crpix = hdulist[0].header['CRPIX1']
	else:
		crpix = 1
	wavelengths = np.arange(fluxes.size)
	wavelengths =  (wavelengths - crpix + 1) * cdelt + crval
	if 'DC-FLAG' in hdulist[0].header and hdulist[0].header['DC-FLAG']:
		#log dispersion
		wavelengths = np.exp(wavelengths * np.log(10))
	hdulist.close()

	return (wavelengths, fluxes, errors)
	

def getspectrum(name, extension=0, axis=0):
	
	return load(name, extension, axis)

def write(name, wavelengths, fluxes, clobber=False):
	hdu = fits.PrimaryHDU(fluxes)
	hdu.header.set('CRPIX1', 1)
	hdu.header.set('CRVAL1', wavelengths[0])
	hdu.header.set('CDELT1', wavelengths[1] - wavelengths[0])
	hdu.header.set('DC-FLAG', 0)
	hdulist = fits.HDUList([hdu])
	
	hdulist.writeto(name, clobber=clobber)
	hdulist.close()

def writespectrum(name, wavelengths, fluxes, clobber=False):
	
	writespectrum(name, wavelengths, fluxes, clobber=False) 

def save(name, wavelengths, fluxes, clobber=False):
	
	writespectrum(name, wavelengths, fluxes, clobber=False)

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
	hdulist = fits.open(name, mode='update')
	header = hdulist[0].header
	for key in pairs.keys():
		header.set(key, pairs[key])
	hdulist.close()

def listheader(name, keys=None):
	hdulist = fits.open(open(os.path.abspath(os.path.expanduser(name)), 'rb'))
	header = hdulist[0].header
	hdulist.close()
	if keys is None:
		return header
	elif isinstance(keys, str):
		return header[keys]
	else:
		output = []
		
		for key in keys:
			output.append(header[key])
		return output
