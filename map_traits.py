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

import pdb
from map_traits_helper import *
import numpy as np
import ray
from spectral.io import envi
import time
import multiprocessing as mp
#import numba
from prosail2 import Prosail
from tqdm import tqdm
import logging
import time
import math

class MapTraits():
    def __init__(self, cnfg):
        # get the cnfg items out into self
        self.input_rfl = cnfg['input']['input_rfl']
        self.input_obs = cnfg['input']['input_obs']
        self.output_dir = cnfg['output']['output_folder']

        # we don't have Sy yet
        # we need to know how many lines and how many rows

        rfl_ds = envi.open(self.input_rfl)
        mmap = rfl_ds.open_memmap()
        # better this way to avoid interleave shananigens
        self.max_y = mmap.shape[0]
        self.max_x = mmap.shape[1]
        self.nbands = mmap.shape[2]
        # we actually do need to know wvs and FWHM

        #obj = rfl_ds.bands
        
        #attributes = [name for name in dir(obj) if not callable(getattr(obj, name)) and not name.startswith('__')]
        wvs = rfl_ds.bands.centers.copy()
        try:
            fwhm = rfl_ds.bands.bandwidths.copy()
        except:
            fwhm = np.ones(len(wvs)) * 20

        self.bands = {'wvs': wvs, 'fwhm': fwhm}

        band_names = ['N-structure', 'Cab', 'Car', 'Cbrown', 'Cw', 'Cm', 'LAI', 'Psoil', \
            'Hspot', 'Leaf Average Slope', 'LIDF bimodality']

        self.out_band_names = band_names + [name + ' uncert' for name in band_names]

        # arrange output file
        # we will put the output files in the output directory
        # for now with the same name + _ls8
        fid = self.input_rfl
        # Regular expression patter
        fid = fid.split('/')[-1].split('.')[0].split('_')[0]

        outfid = self.output_dir + '/' + fid+'_'+'traits.hdr'
        self.output_file = outfid


        # build empty file
        ###################################
        meta = rfl_ds.metadata.copy()
        #meta = {}
        meta['samples'] = self.max_x
        meta['lines'] = self.max_y

        meta['data type'] = 4
        meta.pop('wavelength')
        meta['interleave'] = 'bil'
        description_txt = 'Traits estimation using PROSAIL'
        meta['description'] = description_txt

        meta['band names'] = self.out_band_names
        meta['bands'] = len(self.out_band_names)

        envi.create_image(self.output_file, ext='dat', metadata=meta, force=True)
        ###################################
        # resample prosail dataset file
        # the thing here is that we might be modeling differnet instruments
        # so first we resample and then let prosail use the resampled database
        # so it's also faster
        # we are resampling the database
        dbase_prosail = cnfg['prosail_database_file']
        #pdb.set_trace()
        wvs_dict = self.bands.copy()
        resample_prosail_database(dbase_prosail, wvs_dict)
        #resample_by_instrument_name(dbase_prosail, 'ls8')

    def run(self, localmode=True):
        #pdb.set_trace()

        if localmode is True:
            rayargs = {'local_mode':True, 'num_cpus':1, 'logging_level': logging.ERROR}
            total_cpus = 1
        else:
            rayargs = {'local_mode':False, 'num_cpus': 35, 'logging_level': logging.ERROR}
            total_cpus = 35#mp.cpu_count()
            #rayargs = {'address': 'auto'}
            
        
        # reset ray
        ray.shutdown()
        #pdb.set_trace()
        ray.init(**rayargs)
        #cluster_resources = ray.cluster_resources()
        #total_cpus = int(cluster_resources['CPU'])
        
        print(f"Total number of CPUs in the cluster: {total_cpus}")
        #pdb.set_trace()
        ll = [np.arange(0,self.max_y-1,total_cpus)]
        org = [np.array(ll) + j for j in range(total_cpus)]
        # do the same trick as in geometry to avoid going over the lines limit!!
        
        
        

        jobs = [self.apply_to_row.remote(self, lineset) for lineset in org]
        rreturn = [ray.get(jid) for jid in jobs]


    def run2(self, localmode=True):

        if localmode is True:
            rayargs = {'local_mode': True, 'num_cpus': 1, 'logging_level': logging.ERROR}
        else:
            rayargs = {'local_mode': False, 'num_cpus': mp.cpu_count(), 'logging_level': logging.ERROR}

        ll = [np.arange(0, self.max_y-1, rayargs['num_cpus'])]
        org = [np.array(ll) + j for j in range(rayargs['num_cpus'])]
        # ...

        # reset ray
        ray.shutdown()
        ray.init(**rayargs)

        jobs = [self.apply_to_row.remote(self, lineset) for lineset in org]
        #pdb.set_trace()
        # Wrap with tqdm for progress indication
        with tqdm(total=len(jobs), desc="Processing Rows", unit="row") as pbar:
            for jid in jobs:
                ray.get(jid)
                pbar.update(1)
        print("All rows processed.")

    @ray.remote
    def apply_to_row(self, lineset):#line_start, line_stop):
        # initialize prosail
        
        p = Prosail()
        # run once for numba
        Spherical = (-0.35, -0.15)
        cnf_template = {'N': 1.5, 'Cab': 45, 'Car': 15, 'Cbrown': 0.2, 'Cw': 0.03, \
            'Cm': 0.02, 'LAI': 2, 'psoil': 0.2, 'hspot': 0.5, 'tts': 40, 'tto': 1, 'psi': 60, 'LIDFa': 0, 'LIDFb': 0}

        spc = p.run(cnf_template)

        rfl_ds = envi.open(self.input_rfl)
        
        
        if self.input_obs=='': # i.e., there's no obs file
            noobs_file = True
            band_names = ['To-sun zenith (0 to 90 degrees from zenith)', 'To-sun azimuth (0 to 360 degrees CW from N)', 
            'To-sensor azimuth (0 to 360 degrees CW from N)', 'To-sensor zenith (0 to 90 degrees from zenith)']
        else:
            obs_ds = envi.open(self.input_obs)
            band_names = obs_ds.metadata['band names']
            noobs_file = False

        try:
            rfl_memmap = rfl_ds.open_memmap(writable=False, interleave='bip')
            if noobs_file is False:
                obs_memmap = obs_ds.open_memmap(writable=False, interleave='bip')

            for line in lineset[0]:

                #print(line)
                # hand stop
                #line = 444

                rfl = rfl_memmap[line]
                if noobs_file is False:
                    obs = obs_memmap[line]

                traits_out = []


                for col in range(rfl.shape[0]): # really it could be self.ncols
                    #pdb.set_trace()
                    # hand stop
                    #col = 340
                    #start_time = time.time()
                    print(col)
                    rfl_px = rfl[col]
                    if rfl_px[-1]>3:
                        rfl_px = rfl_px/1000
                    if noobs_file is False:
                        #pdb.set_trace()
                        obs_px = obs[col]
                    else:
                        obs_px = [30, 180, 0, 0]
                    # Radiance
                    wavelengths = self.bands
                    #pdb.set_trace()
                    # I'm passing p, the initialized and ran once to compile with numba ProSAIL model
                    # my hope that it works well with ray and dispatches this object to all the different 
                    # worker nodes and that it is more efficient this way.
                    #pdb.set_trace()
                    if True:#col<4800:

                        traits_est = estimate_traits(rfl_px, obs_px, band_names, wavelengths, p)

                    #else:
                    #    traits_est = [math.nan for _ in range(22)]# from last time it ran

                    traits_out.append(traits_est)

                    #end_time = time.time()  # End timer
                    #elapsed_time = np.round(end_time - start_time, 2)
                    #print(f"Elapsed time for column {col}: {elapsed_time} seconds")
                '''if np.any(np.isfinite(np.asarray(traits_out))):
                    pdb.set_trace()'''
                self.write_to_envi(line, traits_out)

        except Exception as e:
            print("An error occurred:", e)
    
    def write_to_envi(self, line, vals):
        nvalid = len(np.argwhere(np.isfinite(np.asarray(vals))))
        print(f"writing to file line number {line} with {nvalid} valid pixels")
        outname = self.output_file
        out_ds = envi.open(outname)
        out_memmap = out_ds.open_memmap(writable=True, interleave='bip')
        out_memmap[line,:,:] = np.squeeze(np.array(vals))

        del out_memmap
        del out_ds
