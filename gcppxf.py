#! /usr/bin/env python

'''
============================================
Calculate the strength of the CaT using pPXF
============================================

This code uses the method of Foster+ 2010 to measure the strength of the
near infrared calcium triplet (CaT). A description of technique as implemented
is given in Usher+ 2012. The code masks spectral regions affected by sky
emission lines then uses the pPXF code of Cappellari & Emsellem 2004 to fit a
linear combination of template stars to observed spectra. The fitted spectrum
is continuum normalised and the CaT index measured on it. Monte Carlo
resampling can be used to estimate a confidence interval of the index
measurement. Besides measuring the CaT strength, the code also measures the 
radial velocity and velocity dispersion of the spectrum 


Unless otherwise noted wavelengths are in Angstroms and velocities are in km/s.

If you use this code in any publications please cite Usher+ 2012


References
----------

Cappellari, M; Emsellem E; 2004, PASP, 116, 138
Foster, C; Forbes, D A; Proctor, R M; et al. 2010, AJ, 139, 1566
Usher, C; Forbes, D A; Brodie, J P; et al. 2012, MNRAS, 426, 1475


Copyright
---------

Copyright (c) 2014, Christopher Usher
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Christopher Usher nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL CHRISTOPHER USHER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np
import os
import cPickle as pickle
import datetime
import glob
import onedspec
import indices
import normalisation
import interp
import scipy.constants as constants
import gcppxfdata
import multiprocessing
import boot
import ppxf



sky_mask = np.array([[8413, 8418], [8428, 8438],
[8450, 8455], [8463, 8468], [8492, 8496], [8502, 8507], [8536, 8541], [8546,
8551], [8586, 8591], [8595,8599], [8610, 8687], [8757, 8781], [8823, 8839],
[8848, 8852], [8865,8870], [8882, 8888], [8900, 8906], [8917, 8923], [8941,
8946], [8955, 8961]])
'''
Sky line wavelength mask
'''


def cattoz(cat):
    '''
    Convert a CaT index value into metallicity using the conversion of Usher+ 2012
    '''
    
    return np.poly1d([0.46058111, -3.75039355])(cat)



def ppxf_CaT(wavelengths, templates, fluxes, sigmas, vel_scale, delta_v, start_rv, start_sigma, goodpixels, quiet):
    '''
    Run pPXF on a given spectrum
    
    This function runs pPXF on a spectrum using the given templates and returns the
    fitted spectrum, radial velocity, velocity dispersion and CaT index strength.  
    
    Parameters
    ----------
    
    wavelengths : numpy array
        Linearly dispersed input wavelengths
        
    templates : numpy array
        Logarithmically dispersed fluxes of templates. The shape of the array is
        (M, N) where M is number of pixels in a template and N is the number of
        templates 
    
    fluxes : numpy array
        Logarithmically dispersed input fluxes
        
    sigmas : numpy array
        Logarithmically dispersed input sigmas
        
    vel_scale : float
        Natural logarithmic dispersion converted into velocity
        
    delta_v : float
        Difference in natural logarithmic starting wavelengths of the input spectrum
        and the templates in velocity
    
    start_rv : float
        Starting radial velocity for fit
        
    start_sigma : float
        Starting velocity dispersion for fit
        
    goodpixels : numpy array
        Indices of pixels to be fit 
        
    quite : bool
        Whether pPXF should be verbose or not
        
    
    
    Returns
    -------
        
    log_output_fluxes : numpy array
        Logarithmically dispersed fitted fluxes
    
    output_wavelengths : numpy array
        Linearly dispersed fitted wavelengths
        
    output_fluxes : numpy array
        Linear dispersed fitted fluxes
        
    normed_fluxes : numpy array
        Continuum normalised fluxes
        
    rv : float
        Fitted radial velocity
        
    sigma : float
        Fitted velocity dispersion
        
    CaT : float
        Measured CaT strength
    
    '''
    output = ppxf.ppxf(templates, fluxes, sigmas, vel_scale, moments=2, mdegree=7, degree=-1, clean=True, vsyst=delta_v, start=[start_rv, start_sigma], quiet=quiet, goodpixels=goodpixels)

    #convert fitted spectra to linear dispersion and de-redshift
    log_output_fluxes = output.bestfit
    output_wavelengths, output_fluxes = interp.logtolinear(wavelengths, log_output_fluxes, function='nearest', ratio=True)
    rv = output.sol[0]
    sigma = output.sol[1]
    zp1 = 1 + rv * 1e3 / constants.c
    output_wavelengths = output_wavelengths / zp1

    #normalise fitted spectrum and measure CaT strength
    normed_fluxes = normalisation.normalise(output_wavelengths, output_fluxes, 8, normalisation.newmask, True, .004, .010)
    CaT = indices.CaT_gc(output_wavelengths, normed_fluxes, normalized=True)[0]

    return log_output_fluxes, output_wavelengths, output_fluxes, normed_fluxes, rv, sigma, CaT

def _boot_ppxf_CaT(sample):
    '''
    Convenience function for multiprocessing
    '''
    return (ppxf_CaT(*sample))

def ppxfmontecarlo(datum, wavelengths=None, fluxes=None, ivars=None, outputdir='./', nsimulations=0, inversevariance=True, mask=sky_mask, verbose=1, plot=False, rv=0, ident='', keep=True, template_files=None):
    '''
    Measure CaT strength and optionally calculate uncertainty
    
    This function measures the CaT index strength by masking sky lines, fitting a
    linear combination of template spectra using pPXF, normalising the fitted
    spectrum, and measuring the CaT index strength on the normalised spectrum. pPXF
    also fits the radial velocity and velocity dispersion. The function can also
    compute confidence intervals on the CaT strength, radial velocity and velocity
    dispersion by using the fitted spectrum and the error array to generating a 
    series of Monte Carlo realisations of the input spectrum and repeating the 
    measurement process for each. 
    
    
    
    '''
        
    if datum is None:
        
        datum = gcppxfdata.gcppxfdata()
        datum.rv = rv
        
        if ident == '':
            ident = str(np.random.randint(1e10))
        datum.id = ident
        datum.file = ident
        
    start = datetime.datetime.now()

    if wavelengths is None or fluxes is None:
        wavelengths, fluxes = onedspec.getspectrum(datum.scispectrum)
        
    zp1 = 1 + datum.rv * 1e3 / constants.c        
    catrange = (wavelengths > 8425 * zp1) & (wavelengths < 8850 * zp1)
    
    if ivars is None:
        wavelengths, sigmas = onedspec.getspectrum(datum.errorspectrum)
    else:
        sigmas = ivars
        
    if inversevariance:
        np.putmask(sigmas, sigmas <= 0, 1e-10)
        sigmas = sigmas**-.5

    cat_wavelengths = wavelengths[catrange]
    cat_fluxes = fluxes[catrange]        
    datum.catwavelengths = cat_wavelengths
    datum.catfluxes = cat_fluxes
    cat_sigmas = sigmas[catrange]
    datum.catsigmas = cat_sigmas


    s2nregion = (wavelengths > 8400) & (wavelengths < 8500)
    means2n = (fluxes[s2nregion] / sigmas[s2nregion]).mean()
    datum.s2n = means2n / (wavelengths[1] - wavelengths[0])**.5

    log_wavelengths, log_fluxes = interp.lineartolog(cat_wavelengths, cat_fluxes, function='nearest', ratio=True)
    log_wavelengths, log_sigmas = interp.lineartolog(cat_wavelengths, cat_sigmas, function='nearest', ratio=True)
    

    logDispersion = np.log10(log_wavelengths[1]) - np.log10(log_wavelengths[0])
    
    if template_files is None:
        template_files = glob.glob(os.path.expanduser('~') + '/stdstars/t*.fits')
        
        
    if not len(template_files):
        raise Exception('No templates')
    
    log_templates = []
    for template_file in template_files:
        
        template_wavelengths, template_fluxes = onedspec.getspectrum(template_file)
        log_template_wavelengths, log_template_fluxes = interp.lineartolog(template_wavelengths, template_fluxes, function='nearest', ratio=True, logDispersion=logDispersion)
        log_templates.append(log_template_fluxes)
        
    log_templates = np.vstack(log_templates).T
    delta_v = (np.log(log_template_wavelengths[0]) - np.log(log_wavelengths[0])) * constants.c / 1e3


    regionmask = np.ones(log_wavelengths.size, dtype=np.bool_)
    if mask is not None:
        for maskregion in mask:
            regionmask = regionmask & ~((log_wavelengths > maskregion[0]) & (log_wavelengths < maskregion[1]))
        goodpixels = np.nonzero(regionmask)[0]
    else:
        goodpixels = np.arange(log_wavelengths.size)
    
    if verbose > 1:
        quiet = False
    else:
        quiet = True
    
    vel_scale = logDispersion * np.log(10) * constants.c / 1e3
    log_fit_fluxes, datum.fitwavelengths, datum.fitfluxes, datum.normfluxes, datum.fitrv, datum.fitsigma, datum.CaT = ppxf_CaT(log_wavelengths, log_templates, log_fluxes, log_sigmas, vel_scale, delta_v, datum.rv, 10, goodpixels, quiet)

    if plot:
        import matplotlib.pyplot as plt
        
        plt.figure()
        plt.title(datum.file)
        
        plt.plot()
        plt.plot(datum.catwavelengths / (1 + datum.fitrv * 1e3 / constants.c), datum.catfluxes, 'k-')
        plt.plot(datum.fitwavelengths, datum.fitfluxes, 'r-', lw=1.5)
#        residules = datum.catfluxes - datum.fitfluxes
#        plt.plot(datum.fitwavelengths, residules / datum.catsigmas)
#        plt.plot(datum.catwavelengths / (1 + datum.fitrv * 1e3 / constants.c), datum.catsigmas)

        plt.xlabel(u'Wavelength (\u00C5)')
        plt.ylabel('Counts')
    
    datum.samplewavelengths = datum.fitwavelengths
    
    if nsimulations:
    
        samples = []
        for i in range(nsimulations):
            noise = log_sigmas * np.random.normal(size=log_sigmas.size)
            sample_flux = log_fit_fluxes + noise
            
            samples.append((log_wavelengths, log_templates, sample_flux, log_sigmas, vel_scale, delta_v, datum.rv, 10, goodpixels, quiet))
        
        workers = min(multiprocessing.cpu_count(), 12)
        pool = multiprocessing.Pool(processes=workers)        
    
        if verbose > 1:
            print 'Using', workers, 'workers'
            
        sample_results = pool.map(_boot_ppxf_CaT, samples)
        
        sample_fit_fluxes = []
        sample_normed_fluxes = []
        sample_rvs = []
        sample_sigmas = []
        sample_CaTs = []
        
        for i in range(len(sample_results)):
            sample_result = sample_results[i]
            sample_fit_fluxes.append(sample_result[2])
            sample_normed_fluxes.append(sample_result[3])
            sample_rvs.append(sample_result[4])
            sample_sigmas.append(sample_result[5])
            sample_CaTs.append(sample_result[6])
        
        sample_fit_fluxes = np.vstack(sample_fit_fluxes)
        sample_normed_fluxes = np.vstack(sample_normed_fluxes)
        sample_rvs = np.asarray(sample_rvs)
        sample_sigmas = np.asarray(sample_sigmas)
        sample_CaTs = np.asarray(sample_CaTs)
        
        print sample_fit_fluxes.shape

        for i in range(sample_normed_fluxes.shape[1]):
            sample_fit_fluxes[:,i].sort()
            sample_normed_fluxes[:,i].sort()
        lowindex = int(.16 * nsimulations) 
        highindex = int(.84 * nsimulations) - 1

        datum.lowsample = sample_fit_fluxes[lowindex]
        datum.highsample = sample_fit_fluxes[highindex]
        datum.lownormsample = sample_normed_fluxes[lowindex]
        datum.highnormsample = sample_normed_fluxes[highindex]
        
        datum.CaT_samples = sample_CaTs

        rv_peak, rv_lower, rv_upper, rv_spline = boot.kde_interval(sample_rvs)
        datum.fitrvle = datum.fitrv - rv_lower
        datum.fitrvue = rv_upper - datum.fitrv

        sigma_peak, sigma_lower, sigma_upper, sigma_spline = boot.kde_interval(sample_sigmas)
        datum.fitsigmale = datum.fitsigma - sigma_lower
        datum.fitsigmaue = sigma_upper - datum.fitsigma

        CaT_peak, CaT_lower, CaT_upper, CaT_spline = boot.kde_interval(sample_CaTs)
        datum.CaTle = datum.CaT - CaT_lower
        datum.CaTue = CaT_upper - datum.CaT



        if plot:
            plt.figure()
            plt.title(datum.file)
            plt.hist(sample_rvs, histtype='step', normed=True, bins=int(2 * sample_rvs.size**0.5))        
            rvs = np.linspace(2 * sample_rvs.min() - sample_rvs.mean(), 2 * sample_rvs.max() - sample_rvs.mean(), 512)
            plt.plot(rvs, rv_spline(rvs))        
            plt.xlabel(u'rv (km s$^{\\mathregular{\u22121}}$)')
            print 'rv', round(datum.fitrv, 1), round(-datum.fitrvle, 1), round(datum.fitrvue, 1)
    
            plt.figure()
            plt.title(datum.file)            
            plt.hist(sample_sigmas, histtype='step', normed=True, bins=int(2 * sample_sigmas.size**0.5))
            sigmas = np.linspace(2 * sample_sigmas.min() - sample_sigmas.mean(), 2 * sample_sigmas.max() - sample_sigmas.mean(), 512)
            plt.plot(sigmas, sigma_spline(sigmas))
            plt.xlabel(u'\u03C3 (km s$^{\\mathregular{\u22121}}$)')
            print 'sigma', round(datum.fitsigma, 1), round(-datum.fitsigmale, 1), round(datum.fitsigmaue, 1)
    
            plt.figure()
            plt.title(datum.file)            
            plt.hist(sample_CaTs, histtype='step', normed=True, bins=int(2 * sample_CaTs.size**0.5))
            CaTs = np.linspace(2 * sample_CaTs.min() - sample_CaTs.mean(), 2 * sample_CaTs.max() - sample_CaTs.mean(), 512)
            plt.plot(CaTs, CaT_spline(CaTs))
            plt.xlabel(u'CaT (\u00C5)')
            print 'CaT', round(datum.CaT, 2), round(-datum.CaTle, 2), round(datum.CaTue, 2)
            print '[Z/H]', round(datum.CaT * 0.46058111 - 3.75039355, 2), round(-datum.CaTle * 0.46058111, 2), round(datum.CaTue * 0.46058111, 2)

    
    
    else:
        datum.CaTle = 0
        datum.CaTue = 0
        
        datum.lowsample = np.zeros(datum.samplewavelengths.size)
        datum.highsample = np.zeros(datum.samplewavelengths.size)
        datum.lownormsample = np.zeros(datum.samplewavelengths.size)
        datum.highnormsample = np.zeros(datum.samplewavelengths.size)
            
                
    datum.restframemask = sky_mask / zp1
    
    end = datetime.datetime.now()
    datum.runtime = end - start    
    if verbose:
        print outputdir + datum.file + '.pickle'
        print round(datum.CaT, 3), round(datum.CaTle, 3), round(datum.CaTue, 3), round(datum.s2n, 1), datum.runtime, datum.runtime.total_seconds() / (nsimulations + 1)
        print    
    if keep:    
        pickle.dump(datum, open(outputdir + datum.file + '.pickle', 'w'))
    if plot:    
        plt.show()
    return datum    

def test():

    None

