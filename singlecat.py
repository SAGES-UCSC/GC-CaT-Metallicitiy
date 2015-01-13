#! /usr/bin/env python 
'''
Created on May 9, 2011

@author: Chris Usher
'''

import gcppxf
import gcppxfdata
import argparse
import os

parser = argparse.ArgumentParser(description='Measure the CaT Metallicity of a Single Globular Cluster')
parser.add_argument('spectrum', nargs=1, help='spectrum file')
parser.add_argument('error', nargs=1, help='error file')
parser.add_argument('-v', '--rv', type=float, help='radial velocity')
parser.add_argument('-N', '--num-simulations', type=int, default=0, help='Number of Monte Carlo simulations')
parser.add_argument('-o', '--output-dir', help='output directory')
#parser.add_argument('-l', '--idl', default='64', help='IDL version')
parser.add_argument('-s', '--sigma', action='store_true', default=False, help='Error array is sigmas rather than ivars')
parser.add_argument('-n', '--name', help='object name')
parser.add_argument('-p', '--plot', action='store_true', default=False, help='Plot fitted spectra and Monte Carlo results')
parser.add_argument('--colour')
parser.add_argument('--coloure')
parser.add_argument('--radius')



if __name__ == "__main__":
    args = parser.parse_args()
    
    if not args.output_dir:
        args.output_dir = os.getcwd() + '/'
    if not args.name:
        args.name = args.spectrum[0]
    if not args.rv:
        args.rv = 0
    
    datum = gcppxfdata.gcppxfdata()
    datum.id = args.name
    datum.file = args.name
    datum.rv = args.rv
    datum.scispectrum = args.spectrum[0]
    datum.errorspectrum = args.error[0]
    
    gcppxf.ppxfmontecarlo(datum, outputdir=args.output_dir, nsimulations=args.num_simulations, inversevariance=(not args.sigma), plot=args.plot)