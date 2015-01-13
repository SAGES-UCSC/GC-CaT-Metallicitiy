#! /usr/bin/env python
'''
Created on Feb 18, 2011

@author: Chris Usher
'''

import numpy as np

class gcppxfdata:

    def __init__(self):
        
        
        self.id = ''
        self.file = ''
        
        self.scispectrum = ''
        self.errorspectrum = ''        
        
        self.rv = 0
        self.rve = 0
        self.ra = 0
        self.dec = 0
        self.r = 0
        self.colour = 0
        self.coloure = 0
        self.mag = 0
        self.mage = 0
        
        self.fitrv = 0
        self.fitsigma = 0
        
        self.s2n = 0
        
        self.CaT = 0
        self.CaTle = 0
        self.CaTue = 0
        
        self.catwavelengths = np.array([])
        self.catfluxes = np.array([])
        self.catsigmas = np.array([])
        self.fitwavelengths = np.array([])
        self.fitfluxes = np.array([])
        self.normfluxes = np.array([])
        self.samplewavelengths = np.array([])
        self.lowsample = np.array([])
        self.lownormsample = np.array([])
        self.highsample = np.array([])
        self.highnormsample = np.array([])
        self.CaT_samples = np.array([])
        
        self.restframemask = np.array([])
        self.runtime = 0
        