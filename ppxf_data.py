#! /usr/bin/env python

'''
LICENSE
-------------------------------------------------------------------------------
Copyright (c) 2015, Christopher Usher
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-------------------------------------------------------------------------------

Defines the ppxf_data class and provides functions for creating ppxf_data
objects from spectra



'''
 
import numpy as np
import onedspec
import astropy.coordinates


'''
Class for the inputs and outputs of run_ppxf
'''
class ppxf_data():
    
    '''
    Constructor
    
    wavelengths : wavelengths of input spectrum
    fluxes : flux of the input spectrum
    sigmas : 1 sigma uncertainty on input flux
    ident : object identifier
    filename : file that is the source of the input spectrum
    rv_prior : prior on the radial velocity
    sigma_prior : prior on the velocity dispersion
    h3_prior : prior on the h3 moment
    h4_prior : prior on the h4 moment 
    '''
    def __init__(self, wavelengths, fluxes, sigmas, ident='', filename='', rv_prior=0, sigma_prior=10, h3_prior=0, h4_prior=0):
        
        self.ident = ident 
        self.filename = filename
        
        self.rv_prior = rv_prior
        self.sigma_prior = sigma_prior
        self.h3_prior = h3_prior
        self.h4_prior = h4_prior
        
        self.input_wavelengths = wavelengths
        self.input_fluxes = fluxes
        self.input_sigmas = sigmas
        
        
        
'''
Create a ppxf_data object from a fits file or two

fluxes_file : the input flux
sigmas_file : the uncertainties (optional)
ident : the object identifier
filename : file that is the source of the input spectrum. defaults to the value of fluxes_file
ivars  

'''
def create_from_fits(fluxes_file, sigmas_file=None, ident='', filename=None, ivars=True, varis=False, rv_prior=0, sigma_prior=10, h3_prior=0, h4_prior=0):
    
    
    try:
        wavelengths, fluxes, sigmas = onedspec.load_with_errors(fluxes_file)
        
    except IndexError:
        wavelengths, fluxes = onedspec.load(fluxes_file)
    
        if sigmas_file != None:
            wavelengths, sigmas = onedspec.load(sigmas_file)
            if ivars:
                np.putmask(sigmas, sigmas <= 0, 1e-10)
                sigmas = sigmas**-0.5
            elif varis:
                sigmas = sigmas**0.5     
        else:
            sigmas = np.ones(fluxes.size) * fluxes.mean()
    
    
    try:        
        raw_ra, raw_dec = onedspec.listheader(fluxes_file, ['RA', 'DEC'])
        coordinates = astropy.coordinates.SkyCoord(raw_ra, raw_dec, unit=(astropy.units.hourangle, astropy.units.deg))
        ra = coordinates.ra.deg
        dec = coordinates.dec.deg
         
    except KeyError:
        ra = None
        dec = None
    
    if filename == None:
        filename = fluxes_file
    
    datum = ppxf_data(wavelengths, fluxes, sigmas, ident=ident, filename=filename, rv_prior=rv_prior, sigma_prior=sigma_prior, h3_prior=h3_prior, h4_prior=h4_prior)
    datum.ra = ra
    datum.dec = dec
    
    return datum

'''
Create a ppxf_data object from an ascii file
'''
def create_from_ascii(ascii_file, ident='', filename=None, ivars=False, varis=False, rv_prior=0, sigma_prior=10, h3_prior=0, h4_prior=0):
    wavelengths = []
    fluxes = []
    sigmas = []
    
    columns = 0
    
    lines = open(ascii_file).readlines()
    for line in lines:
        
        line = line.strip()
        if line == '' or line[0] == '#':
            continue
        
        substrs = line.split()
        
        if not columns:
            columns = len(substrs)
            if columns < 2:
                raise Exception('Too few columns')
            
        elif columns != len(substrs):
            raise Exception('Inconsistent number of columns')
        
        wavelengths.append(float(substrs[0]))
        fluxes.append(float(substrs[1]))
        if columns == 3:
            sigmas.append(float(substrs[2]))
    
    wavelengths = np.array(wavelengths)
    fluxes = np.array(fluxes)
    if columns == 3:
        sigmas = np.array(sigmas)
    else:
        sigmas = np.ones(fluxes.size)

    if ivars:
        np.putmask(sigmas, sigmas <= 0, 1e-10)
        sigmas = sigmas**-0.5
    elif varis:
        sigmas = sigmas**0.5

    if filename == None:
        filename = ascii_file
    
    datum = ppxf_data(wavelengths, fluxes, sigmas, ident=ident, filename=filename, rv_prior=rv_prior, sigma_prior=sigma_prior, h3_prior=h3_prior, h4_prior=h4_prior)
    return datum
    