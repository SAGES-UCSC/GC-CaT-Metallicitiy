#! /usr/bin/env python 
'''
Created on May 28, 2013

@author: Chris Usher
'''

import os


def uopen(name, *args):
    
    return open(os.path.abspath(os.path.expanduser(name)), *args)