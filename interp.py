#! /usr/bin/env python
'''
Created on Mar 17, 2011

@author: Chris Usher
'''

import numpy as np
#import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

def redisperse(inputwavelengths, inputfluxes, firstWavelength=None, lastWavelength=None, dispersion=None, nPixels=None, outside=None, function='spline'):

    inputedges = np.empty(inputwavelengths.size + 1)
    inputedges[1:-1] = (inputwavelengths[1:] + inputwavelengths[:-1]) / 2
    inputedges[0] = 3 * inputwavelengths[0] / 2 - inputwavelengths[1] / 2
    inputedges[-1] = 3 * inputwavelengths[-1] / 2 - inputwavelengths[-2] / 2

    inputdispersions = inputedges[1:] - inputedges[:-1]

    epsilon = 1e-10

    if dispersion == None and nPixels != None:

        if firstWavelength == None:
            firstWavelength = inputwavelengths[0]
        if lastWavelength == None:
            lastWavelength = inputwavelengths[-1]    
        outputwavelengths = np.linspace(firstWavelength, lastWavelength, nPixels)

    elif dispersion != None and nPixels == None:
        if firstWavelength == None:
            firstWavelength = inputwavelengths[0]
        if lastWavelength == None:
            lastWavelength = inputwavelengths[-1] 
        outputwavelengths = np.arange(firstWavelength, lastWavelength + epsilon, dispersion)

    elif dispersion != None and nPixels != None:
        if firstWavelength != None:
            outputwavelengths = firstWavelength + dispersion * np.ones(nPixels)
        elif lastWavelength != None:
            outputwavelengths = lastWavelength - dispersion * np.ones(nPixels)
            outputwavelengths = outputwavelengths[::-1]
        else:
            outputwavelengths = inputwavelengths[0] + dispersion * np.ones(nPixels) 

    else:
        dispersion = (inputwavelengths[-1] - inputwavelengths[0]) / (inputwavelengths.size - 1)
        if lastWavelength == None:
            lastWavelength = inputwavelengths[-1]
        if firstWavelength != None:
            outputwavelengths = np.arange(firstWavelength, lastWavelength + epsilon, dispersion)
        else:
            outputwavelengths = np.arange(inputwavelengths[0], lastWavelength + epsilon, dispersion)

    outputdispersion = outputwavelengths[1] - outputwavelengths[0]
    outputedges = np.linspace(outputwavelengths[0] - outputdispersion / 2, outputwavelengths[-1] + outputdispersion / 2, outputwavelengths.size + 1)

    outputfluxes = interp(inputwavelengths, inputfluxes, inputedges, inputdispersions, outputwavelengths, outputedges, outside, function)

    return (outputwavelengths, outputfluxes)

def rebin(inputwavelengths, inputfluxes, outputwavelengths, outside=None, function='spline', ratio=False):

    inputedges = np.empty(inputwavelengths.size + 1)
    inputedges[1:-1] = (inputwavelengths[1:] + inputwavelengths[:-1]) / 2
    inputedges[0] = 3 * inputwavelengths[0] / 2 - inputwavelengths[1] / 2
    inputedges[-1] = 3 * inputwavelengths[-1] / 2 - inputwavelengths[-2] / 2

    inputdispersions = inputedges[1:] - inputedges[:-1]

    outputedges = np.empty(outputwavelengths.size + 1)
    outputedges[1:-1] = (outputwavelengths[1:] + outputwavelengths[:-1]) / 2
    outputedges[0] = 3 * outputwavelengths[0] / 2 - outputwavelengths[1] / 2
    outputedges[-1] = 3 * outputwavelengths[-1] / 2 - outputwavelengths[-2] / 2

    return interp(inputwavelengths, inputfluxes, inputedges, inputdispersions, outputwavelengths, outputedges, outside, function, ratio)


def interp(inputwavelengths, inputfluxes, inputedges, inputdispersions, outputwavelengths, outputedges, outside=None, function='spline', ratio=False):
    
    
    if not ratio:
        fluxdensities = inputfluxes / inputdispersions.mean()
    else:
        fluxdensities = inputfluxes
    
    outputfluxes = np.ones(outputwavelengths.size)
    
    if outside != None:
        outputfluxes = outputfluxes * outside
    else:
        middle = (outputwavelengths[0] + outputwavelengths[-1]) / 2

    firstnew = None
    lastnew = None
    
    if function == 'nearest':
        pixels = np.arange(0, inputfluxes.size)
        for newpixel in range(outputfluxes.size):

            if inputedges[0] <= outputwavelengths[newpixel] <= inputedges[-1]:
                outputlowerlimit = outputedges[newpixel]
                outputupperlimit = outputedges[newpixel + 1]
                outputfluxes[newpixel] = 0
                
                below = inputedges[1:] < outputlowerlimit
                above = inputedges[:-1] > outputupperlimit
                ok = ~(below | above)
                

                for oldpixel in pixels[ok]:
                    inputlowerlimit = inputedges[oldpixel]
                    inputupperlimit = inputedges[oldpixel + 1]

                    if inputlowerlimit >= outputlowerlimit and inputupperlimit <= outputupperlimit:
                        outputfluxes[newpixel] += fluxdensities[oldpixel] * inputdispersions[oldpixel]

                    elif inputlowerlimit < outputlowerlimit and inputupperlimit > outputupperlimit:
                        outputfluxes[newpixel] += fluxdensities[oldpixel] * (outputupperlimit - outputlowerlimit)

                    elif inputlowerlimit < outputlowerlimit and outputlowerlimit <= inputupperlimit <= outputupperlimit:
                        outputfluxes[newpixel] += fluxdensities[oldpixel] * (inputupperlimit - outputlowerlimit)

                    elif outputupperlimit >= inputlowerlimit >= outputlowerlimit and inputupperlimit > outputupperlimit:
                        outputfluxes[newpixel] += fluxdensities[oldpixel] * (outputupperlimit - inputlowerlimit)
                        
                if firstnew == None:
                    firstnew = outputfluxes[newpixel]
            
                if ratio:
                    outputfluxes[newpixel] = outputfluxes[newpixel] / (outputupperlimit - outputlowerlimit)
                    
            elif outputwavelengths[newpixel] > inputwavelengths[-1] and lastnew == None:
                lastnew = outputfluxes[newpixel - 1]
                
                

    else:
        fluxspline = interpolate.UnivariateSpline(inputwavelengths, fluxdensities, s=0, k=3)
        
        for newpixel in range(outputfluxes.size):
            if inputedges[0] <= outputwavelengths[newpixel] <= inputedges[-1]:
                
                outputlowerlimit = outputedges[newpixel]
                outputupperlimit = outputedges[newpixel + 1]
                
                
                outputfluxes[newpixel] = fluxspline.integral(outputedges[newpixel], outputedges[newpixel + 1])
                if firstnew == None:
                    firstnew = outputfluxes[newpixel]

                if ratio:
                    outputfluxes[newpixel] = outputfluxes[newpixel] / (outputupperlimit - outputlowerlimit)
                    
            elif outputwavelengths[newpixel] > inputwavelengths[-1] and lastnew == None:
                lastnew = outputfluxes[newpixel - 1]    
            
                
        if outside == None:
            for newpixel in range(outputfluxes.size):
                if outputwavelengths[newpixel] < inputwavelengths[0]:
                    outputfluxes[newpixel] = firstnew
                elif outputwavelengths[newpixel] > inputwavelengths[-1]:
                    outputfluxes[newpixel] = lastnew
            
                
    return outputfluxes

def lineartolog(inputwavelengths, inputfluxes, outside=0, function='spline', ratio=False, logDispersion=0):

    inputedges = np.empty(inputwavelengths.size + 1)
    inputedges[1:-1] = (inputwavelengths[1:] + inputwavelengths[:-1]) / 2
    inputedges[0] = 3 * inputwavelengths[0] / 2 - inputwavelengths[1] / 2
    inputedges[-1] = 3 * inputwavelengths[-1] / 2 - inputwavelengths[-2] / 2
    inputdispersions = inputedges[1:] - inputedges[:-1]
    
    if logDispersion:
        outputedges = np.arange(np.log10(inputedges[0]), np.log10(inputedges[-1]), logDispersion)
        outputwavelengths = (outputedges[:-1] + outputedges[1:]) / 2
        
        outputedges = 10**outputedges
        outputwavelengths = 10**outputwavelengths
        
        
    else:
        outputedges = np.logspace(np.log10(inputedges[0]), np.log10(inputedges[-1]), inputedges.size)    
        outputwavelengths = (outputedges[:-1] * outputedges[1:])**.5
    
    return outputwavelengths, interp(inputwavelengths, inputfluxes, inputedges, inputdispersions, outputwavelengths, outputedges, outside, function, ratio)
    
    
def logtolinear(inputwavelengths, inputfluxes, outside=0, function='spline', ratio=False):
    
    logWavelengths = np.log10(inputwavelengths)
    
    inputedges = np.empty(logWavelengths.size + 1)
    inputedges[1:-1] = (logWavelengths[1:] + logWavelengths[:-1]) / 2
    inputedges[0] = 3 * logWavelengths[0] / 2 - logWavelengths[1] / 2
    inputedges[-1] = 3 * logWavelengths[-1] / 2 - logWavelengths[-2] / 2

    inputedges = 10**inputedges
    inputdispersions = inputedges[1:] - inputedges[:-1]
    outputedges = np.linspace(inputedges[0], inputedges[-1], inputedges.size)
    outputwavelengths = (outputedges[:-1] + outputedges[1:]) / 2
    
    return  outputwavelengths, interp(inputwavelengths, inputfluxes, inputedges, inputdispersions, outputwavelengths, outputedges, outside, function, ratio)
    
    


#plt.show()
