#! /usr/bin/env python

'''
LICENSE
-------------------------------------------------------------------------------
Copyright (c) 2015 to 2016, Christopher Usher
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

Code to run ppxf using input from the command line

Will accept pickled ppxf_data objects (as .ppxf), fits spectra and ascii spectra as input

'''
 
import argparse
import pickle
import run_ppxf
import ppxf_data


parser = argparse.ArgumentParser(description='Run pPXF on a single spectrum and optionally use Monte Carlo simulations to estimate the uncertainty on the fit')
parser.add_argument('input', help='Input file. If file ends in .fits, the code assumes it is a fits file. If it ends in .ppxf, it assumes it is a pickled ppxf_data object. Otherwise, the code assumes that it is ascii')
parser.add_argument('errors', nargs='?', help='Error file. Only used if input file is a fits file')

error_type = parser.add_mutually_exclusive_group()
error_type.add_argument('--ivars', action='store_true', help='Uncertainty is stored as inverse variances')
error_type.add_argument('--varis', action='store_true', help='Uncertainty is stored as variances')


parser.add_argument('-v', '--rv', type=float, default=0, help='Radial velocity prior')
parser.add_argument('-s', '--sigma', type=float, default=10, help='Velocity dispersion prior')
parser.add_argument('--h3', type=float, default=0, help='h3 prior')
parser.add_argument('--h4', type=float, default=0, help='h4 prior')

parser.add_argument('--order', type=float, default=7, help='Order of the continuum polynomial')
parser.add_argument('--moments', type=float, default=2, help='Velocity moments to fit')

parser.add_argument('-N', '--num-simulations', type=int, default=0, help='Number of Monte Carlo simulations')
parser.add_argument('--quiet', action='store_false', help='Suppress output')

extras = parser.add_mutually_exclusive_group()
extras.add_argument('--CaT', action='store_true', help='Measure CaT using Usher+12 technique')
extras.add_argument('--CaT_C01', action='store_true', help='Measure CaT using Cennaro+01 definition')
extras.add_argument('--kinematics', action='store_true', help='Just measure the kinematics')

parser.add_argument('--no-mask', action='store_true', help='Don\'t mask wavelength ranges with emission lines')


parser.add_argument('--templates', nargs='+', help='Template files.\nUse DEIMOS for the DEIMOS files, Cenarro01 for the Cenarro 2001 library and MILES for the MILES spectral library')
parser.add_argument('--save', action='store_true', help='Save output as pickled ppxf_data object')
parser.add_argument('-n', '--name', default='', help='Object name')
parser.add_argument('-p', '--plot', action='store_true', help='Plot fitted spectra and Monte Carlo results')




if __name__ == "__main__":
    args = parser.parse_args()

#    print args.templates
                
    if args.templates == None:
        print 'Using standard templates'
    elif args.templates[0] in ['DEIMOS', 'deimos']:
        print 'Using the DEIMOS templates'
        templates = run_ppxf.load_CaT_templates
    elif args.templates[0] in ['Cenarro01', 'C01', 'cenarro', 'Cenarro']:
        print 'Using the Cenarro 2001 templates'
        templates = run_ppxf.load_C01_templates
    elif args.templates[0] in ['miles', 'MILES']:
        print 'Using the MILES templates'
        templates = run_ppxf.load_miles_templates
    elif args.templates != None:
        print 'Using user supplied templates'
        templates = args.templates


    if args.input[-5:] == '.ppxf':
        
        input_datum = pickle.load(open(args.input))
        output_name = args.input[:-5]
        args.save = True
        
    elif args.input[-5:] == '.fits':
        
        input_datum = ppxf_data.create_from_fits(args.input, args.errors, ident=args.name, ivars=args.ivars, varis=args.varis,
                                                rv_prior=args.rv, sigma_prior=args.sigma, h3_prior=args.h3, h4_prior=args.h4)
        output_name = args.input[:-5]
    
    #otherwise assume ascii    
    else:
               
        input_datum = ppxf_data.create_from_ascii(args.input, ident=args.name, ivars=args.ivars, varis=args.varis,
                                          rv_prior=args.rv, sigma_prior=args.sigma, h3_prior=args.h3, h4_prior=args.h4)
        output_name = args.input

    if args.no_mask:
        mask = None
    else:
        mask = run_ppxf.CaT_mask        
    
    if args.kinematics:
        if args.templates == None:
            templates = run_ppxf.load_CaT_templates


        output_datum = run_ppxf.ppxf_kinematics(input_datum, templates, nsimulations=args.num_simulations, verbose=args.quiet, plot=args.plot, moments=args.moments, degree=args.order)

        
    elif args.CaT_C01:
        if args.templates == None:
            templates = run_ppxf.load_C01_templates

            
        output_datum = run_ppxf.ppxf_CaT_C01(input_datum, templates, nsimulations=args.num_simulations, verbose=args.quiet, plot=args.plot, mask=mask)


    else:        
        if args.templates == None:
            templates = run_ppxf.load_CaT_templates


        output_datum = run_ppxf.ppxf_CaT(input_datum, templates, nsimulations=args.num_simulations, verbose=args.quiet, plot=args.plot, mask=mask)
        
    if args.save:
        pickle.dump(output_datum, open(output_name + '.ppxfout', 'w'))
        
    if args.plot:
        import matplotlib.pyplot as plt
        plt.show()
            
    
            
            
        
        
        
    
        
        
        
        
    
    

