#!/usr/bin/python
# coding=utf-8
#
#  Copyright 2024 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#
#
# Plant Trait Estimation from Radiance for Fire Risk Estimation
# Author: Nimrod Carmon nimrod.carmon@jpl.nasa.gov


import os
import numpy as np
from numpy import pi
import pdb
from helper2 import *
from types import SimpleNamespace 
import cProfile
import time
import numba
import pdb
import matplotlib.pyplot as plt
numba.set_num_threads(1)  # Enable NUMBA diagnostics

# This is the prosail driver. It mainly grabs stuff from helper2.py and I think also deals with numba and multiple processing


class Prosail():
    # PROSAIL in python
    def __init__(self):
        
        a = 1
        # not sure yet what to put here
        # maybe just the setting up stuff
        # and maybe wavelengths for the target instrument
        # Spectra data import
        self.headers, self.spectra = dataSpec_P5B() # This function is in helper2.py
        #pdb.set_trace()
        self.Rsoil1=np.array(self.spectra[9])# dry soil and wet soil. They are featureless for the most part
        self.Rsoil2=np.array(self.spectra[10])#
        self.Es=np.array(self.spectra[7])# these two are the direct and diffuse fluxes, and are also very general
        self.Ed=np.array(self.spectra[8])#
        self.wl = self.spectra[0]

        # building a hashtable now
        self.cache = {}
        self.precision = 3

    #@numba.njit(nopython=True)
    def run(self, conf: dict):
        
        # rounding for cache
        rounded_conf = {key: round(value, self.precision) if isinstance(value, float) else value for key, value in conf.items()}
        # Convert the rounded configuration to a hashable type
        # I think maybe we should normalize the ranges and have scalar unnormaliztion stuff to go back so then all the quantities are similar 
        hashable_config = tuple(sorted(rounded_conf.items()))
        if hashable_config in self.cache:
            #print('used the hashtable')
            return self.cache[hashable_config]

        else:
            # Unpack the dictionary dynamically
            # this is a non descriptive way to get the key-value pairs our into variables
            # if you really care about it just print the key names
            #pdb.set_trace()
            '''N = rounded_conf['N']
            Cab = rounded_conf['Cab']
            Car = rounded_conf['Car']
            Cbrown = rounded_conf['Cbrown']
            Cw = rounded_conf['Cw']
            Cm = rounded_conf['Cm']
            LAI = rounded_conf['LAI']
            psoil = rounded_conf['psoil']
            hspot = rounded_conf['hspot']
            tts = rounded_conf['tts'] solar zenith
            tto = rounded_conf['tto'] to sensor zenith
            psi = rounded_conf['psi'] relative azimuth angle
            #LIDF = conf['LIDF']
            LIDFa = rounded_conf['LIDFa']
            LIDFb = rounded_conf['LIDFb']'''

            for key, value in conf.items():
                globals()[key] = value

            #pdb.set_trace()

            # first we run prospect:
            # first output is reflectance, second is transmmitance
            # This is the PROSPECT leaf reflectance and transmmitance model
            rho, tau= prospect_5B2(N, Cab, Car, Cbrown, Cw, Cm, self.spectra)
            #pdb.set_trace()
            # now we need to calculate the lead distribution, which is an input to the sail model.

            # Calculate LIDF output
            lidf = calcLidf(LIDFa, LIDFb)

            # get soil mixture. This is literally two soil spectra, one wet one dry, scaled using the psoil parameter
            rsoil0 = psoil * self.Rsoil1 + (1 - psoil) * self.Rsoil2

            # run the canopy model
            rsot, rdot, rsdt, rddt= PRO4SAIL(rho, tau, lidf, LAI, hspot, tts, tto, psi, rsoil0)

            resh, resv = canref(rsot, rdot, rsdt, rddt, self.Es, self.Ed, tts)

            self.cache[hashable_config] = resv

            return resv





