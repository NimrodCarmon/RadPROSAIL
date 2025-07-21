#! /usr/bin/env python3


import glob
import pdb
import numpy as np
from modtran_rt_helper import get_modtran_template_one_part, get_modtran_template_quick_tp6
import json
from run_all_modtran import run_modtran_on_dir as runm
import os
from channel_read import load_chn, load_chn_single
import pickle
import ray
from format_hdf5 import SaveToHDF5, SaveToHDF5_multimodel
from copy import deepcopy
import inspect
from itertools import product
import re
import multiprocessing as mp
from tqdm import tqdm
from scipy.stats.qmc import Sobol
from process_tp7_9case import main as process_tp7
from scipy.stats import qmc
import time

class Modtran_rt():
    """
    This class does the heavy lifting of this codebase.
    It sets up a modtran template, builds json files out of it, then after modtran was ran, it can read and disect the output
    """
    def __init__(self, filter_file, output_dir, ncpus, sample_space, sampling_type, multipart, config):
        '''
        Regular sampling doesn'st just mean to have grid points on a grid, but it means that all the grid points
        must exist. This is a critical distinction for downstream stuff!
        '''
        # grab info and save to self
        #import pdb; pdb.set_trace()
        self.config = config
        self.filter_file = filter_file
        self.output_dir = output_dir
        self.lutdir = self.output_dir
        self.default_angle = 180
        self.universal_lut = []
        self.multipart = multipart
        self.band_model = config['implementation']['band_model']
        self.multiband = config['implementation']['multiband']
        #self.allow_rh_ht_100 = config['MISC']['Allow_RH_gt_100']
        if self.multiband is True:
            self.band_cutoffs = config['implementation']['bandstops']
        try:
            self.n_samples = config['implementation']['n_samples']
        except:
            pass
        # the sorted state vector names list
        # This is important because we are sorting the parameter space by the name!
        self.state_vecs = sorted(sample_space.keys())
        
        if 'batch_dir' in config['output']:
            self.batch_dir = config['output']['batch_dir']

        if False:#config['MISC']['use_user_defined_aerosol'] is True:
            self.use_aerosol_model = True

            tmpl_file = config['MISC']['User_defined_aerosol']['template_file']
            with open(tmpl_file, 'r') as f:
                aerosol_tmpl = json.load(f)
            model_file = config['MISC']['User_defined_aerosol']['model_file']
            aer_data = np.loadtxt(model_file)
            # now read first the wavelengths from the first column
            aer_wl = aer_data[:,0]
            # now, we get the user defined index of the model from the config
            model_index = config['MISC']['User_defined_aerosol']['model_index']
            # now get the information
            start_col = 1+model_index*3 # pythonic counting starts at 0
            aer_extc = aer_data[:, start_col]
            aer_absc = aer_data[:, start_col+1]
            aer_asym = aer_data[:, start_col+2]

            # I think because we're not deep coping we can use user_aerosol_model as a pointer to the template
            user_aerosol_model = aerosol_tmpl['IREGSPC'][0] # just remember it's a list in the config
            user_aerosol_model['NARSPC'] = int(len(aer_wl))
            user_aerosol_model['VARSPC'] = list(aer_wl)
            user_aerosol_model['EXTC'] = list(aer_extc)
            user_aerosol_model['ABSC'] = list(aer_absc)
            user_aerosol_model['ASYM'] = list(aer_asym)

            self.aerosol_tmpl = aerosol_tmpl

            

        # This is for a regular grid
        # If people want to include non regular grid in the future I do have code for it from Jouni's project.
        input_dict = {}
        #pdb.set_trace()
        self.sampling = sampling_type
        regular_grid = (sampling_type=='regular')
        
        if regular_grid is True:
            # For every parameter in the sample_space
            for param, param_info in sample_space.items():
                # let's add an option for we just have a single value for one of the parameters

                param_min = param_info['min']
                param_max = param_info['max']
                if param_min != param_max:
                    param_spacing = param_info['spacing']
                    param_lut = np.arange(param_min, param_max, param_spacing)
                else:
                    # if the min and max are the same, which is a flag for having a single value of the parameter
                    # then we just write it into the dict! hope this works with no bugs..
                    param_lut = [param_min] # this must be a list even if there's just one value because the 'combination' functionality 
                input_dict[param] = param_lut

        elif sampling_type=='sobol':
            for param, param_info in sample_space.items():
                param_min = param_info['min']
                param_max = param_info['max']
                param_lut = np.array([param_min, param_max])
                input_dict[param] = param_lut

        elif sampling_type=='relative_water':
            print('Under Construction')
            # This I think could be either sobol or regular
            # This is a totally new idea, Lex Berk gave a stamp of approval
            # The idea is to calculate the maximum amount of water vapor per ground altitude
            # Then the water vapor amount will be relative to that maximum
            # We don't have the term to use for this still. Lex said 'percent saturation' as a place holder
            # But I'm thinking maybe Atmospheric Moisture Capacity (AMC) or something similar
            # So far there were a bunch of arguments against spanning from 'why do we even need it'
            # IE what kind of problem we are facing that this solves, and than some stuff about the 
            # physics, the optimization, etc., but they will turn around bc I think it is actually 
            # the correct way to go forward here.

            # We will need to get the flag from the config. It's under implementation['CRH'] true\false
            # Then we have to calculate the max value for each gridpoint, right?
            # This should be done for every elevation value. Then for each elevation value we go with the [0,0.1,1]
            # grid.. so this involves running it with only tp6 first and 10 grams, then grabbing the adjusted value
            # then calculating everything else..

            # step 1: run modtran with 20 grams to find max values over surface elevation grid.
            
            for param, param_info in sample_space.items():
                # let's add an option for we just have a single value for one of the parameters
                if param == 'GNDALT':
                    
                    param_min = param_info['min']
                    param_max = param_info['max']
                    param_spacing = param_info['spacing']
                    GNDALT_values = np.arange(param_min, param_max, param_spacing)

            #pdb.set_trace()
            # now we have to construct a json input with the tp6 only option, and read the max value for each GNDALT
            # we only need whatever SZA and aerosols..
            input_dict2 = {}
            input_dict2['H1'] = [config['MISC']['H1']]
            input_dict2['RAA'] = [self.default_angle]
            input_dict2['TSZ'] = [self.default_angle]
            input_dict2['AOT550'] = [0.1]
            input_dict2['GNDALT'] = [0]
            input_dict2['H2OSTR'] = [10]
            input_dict2['SZA'] = [30]
            self.max_per_gndalt = self.calc_max_per_gndalt(GNDALT_values, input_dict2)
            #pdb.set_trace()

            # now we have to do the regular grid, but change the water vapor amount by dividing by max_per_gndalt for the 
            # appropriate gndalt value
            for param, param_info in sample_space.items():
                # let's add an option for we just have a single value for one of the parameters

                param_min = param_info['min']
                param_max = param_info['max']
                if param_min != param_max:
                    param_spacing = param_info['spacing']
                    param_lut = np.arange(param_min, param_max, param_spacing)
                else:
                    # if the min and max are the same, which is a flag for having a single value of the parameter
                    # then we just write it into the dict! hope this works with no bugs..
                    param_lut = [param_min] # this must be a list even if there's just one value because the 'combination' functionality 
                
               
                input_dict[param] = param_lut
                # somewhere we will have to adjust from relative water to absolute so we can plug it into the json file

            

        # fixed parameters
        # at some point think about the values here
        if 'TSZ' in input_dict.keys():
            # we just use the values from the config, adjusted for 180 minus
            print('wait')
            input_dict['TSZ'] = 180-input_dict['TSZ']
        else:
            try:
                #pdb.set_trace()
                input_dict['TSZ'] = [180 - float(config['MISC']['OBSZEN'])]
            except:
                input_dict['TSZ'] = [self.default_angle]

        if 'RAA' in input_dict.keys():
            # use what's in the config, no need to transform
            pass
        else:
            input_dict['RAA'] = [self.default_angle]
        # maybe for airborne LUTs we'll need to do the same for H1, but for now we don't.
        #pdb.set_trace()
        input_dict['H1'] = [config['MISC']['H1']]
        self.input_dict = input_dict
        #pdb.set_trace()


    def calc_max_per_gndalt(self, GNDALT_values, input_dict):
        '''This function will return the max water vapor value per each GNDALT_values entry'''
        lut_names = []
        # constants:
        #pdb.set_trace()
        mod_input = {
            'atmosphere_type': self.config['MISC']['Atmospheric_Profile'],
            'alt': float(input_dict['H1'][0]),
            'aerosol_model': 'AER_RURAL',
            'filter_file': self.filter_file,
            'TSZ': float(input_dict['TSZ'][0]),
            'RAA': float(self.default_angle),
            'band_model': self.band_model
        }

        parameters = self.state_vecs
        
        if True:#self.sampling == 'regular':
            input_dict['GNDALT'] = GNDALT_values.tolist()
            param_values = [input_dict[param] for param in parameters]

            all_combinations = list(product(*param_values))

        output_files = []
        for combination in all_combinations:
            #pdb.set_trace()
            
            mod_input.update({param: float(value) for param, value in zip(parameters, combination)})
            #param = []

            outf = '_'.join(['%s-%6.4f' % (param, mod_input[param]) for param in parameters])
            mod_input['NAME'] = outf
            lut_point_json = get_modtran_template_quick_tp6(mod_input)
            param = lut_point_json["MODTRAN"]
            jsontmpl_11 = deepcopy(lut_point_json["MODTRAN"][0])
            #param.append(jsontmpl_11)
            output_temp_dir = self.output_dir+'/temp/'
            os.makedirs(output_temp_dir, exist_ok=True)

            output_file = output_temp_dir+outf+'.json'
            output_files.append(output_file)
            #pdb.set_trace()
            with open(output_file, 'w') as fout:
                fout.write(json.dumps({"MODTRAN": param}, cls=SerialEncoder, indent=4, sort_keys=True))


        # now we need to run modtran on this
        
        original_dir = os.getcwd()
        os.chdir(output_temp_dir)
        modtran_exe = os.environ['MODTRAN_EXE']
        for fname in output_files:
            cmd = modtran_exe + ' ' + fname
            if os.path.exists(fname):
                pass
            else:
                os.system(cmd)
        os.chdir(original_dir)

        #pdb.set_trace()
        data = extract_water_vapor_data_from_tp6(output_temp_dir)
        sorted_data = data[np.argsort(data[:, 0])]
        return sorted_data
        # now we need to grab the info from the files



    def build_jsons_4tp6(self):
        # a very very light weight configuration that will just give us the tp6 output
        # It will run fast and will have a weird config file.
        # Still it must run on the same input space
        # this is a fix because the earlier runs didn't have tp6 outputs.
         # builds the json files
        lut_names = []
        # constants:
        #pdb.set_trace()
        mod_input = {
            'atmosphere_type': self.config['MISC']['Atmospheric_Profile'],
            'alt': float(self.input_dict['H1'][0]),
            'aerosol_model': 'AER_RURAL',
            'filter_file': self.filter_file,
            'TSZ': float(self.input_dict['TSZ'][0]),
            'RAA': float(self.default_angle),
            'band_model': self.band_model
        }

        parameters = self.state_vecs

        if True:#self.sampling == 'regular':
            param_values = [self.input_dict[param] for param in parameters]
            all_combinations = list(product(*param_values))
            n_lut_build = len(all_combinations)
        
        for combination in all_combinations:
            #pdb.set_trace()
            
            mod_input.update({param: float(value) for param, value in zip(parameters, combination)})
            #param = []

            outf = '_'.join(['%s-%6.4f' % (param, mod_input[param]) for param in parameters])
            mod_input['NAME'] = outf
            lut_point_json = get_modtran_template_quick_tp6(mod_input)
            param = lut_point_json["MODTRAN"]
            jsontmpl_11 = deepcopy(lut_point_json["MODTRAN"][0])
            #param.append(jsontmpl_11)
            output_file = self.output_dir+'/'+outf+'.json'
            with open(output_file, 'w') as fout:
                fout.write(json.dumps({"MODTRAN": param}, cls=SerialEncoder, indent=4, sort_keys=True))



    def build_jsons(self):
        # builds the json files
        lut_names = []
        # constants:
        mod_input = {
            'atmosphere_type': 'ATM_TROPICAL',
            'alt': float(self.input_dict['H1'][0]),
            'aerosol_model': 'AER_RURAL',
            'filter_file': self.filter_file,
            'TSZ': float(self.input_dict['TSZ'][0]),
            'RAA': float(self.default_angle),
            'band_model': self.band_model
        }

        # parameters to iterate over, sorted alphabetically
        # following state vector naming convensions from isofit
        
        parameters = self.state_vecs
        # the idea is that whatever is in the state vector is going to change in the json file
        # but if there are things that are not in the state vectors, then there's a default
        # value that is used.
        
        # get all combinations of parameters
        #pdb.set_trace()
        if self.sampling == 'regular' or self.sampling == 'relative_water':
            param_values = [self.input_dict[param] for param in parameters]
            all_combinations = list(product(*param_values))
            n_lut_build = len(all_combinations)
        elif self.sampling == 'sobol':
            # we might have parameters that after transformation have first a higher value than the second

            # Extract min and max values for each parameter
            #ranges = [(self.input_dict[param][0], self.input_dict[param][1]) for param in parameters]
            ranges = [(np.min(self.input_dict[param]), np.max(self.input_dict[param])) for param in parameters]

            # Create a Sobol sequence sampler
            sampler = qmc.Sobol(d=len(ranges), scramble=True)

            # Generate samples (e.g., 500 samples)
            num_samples = self.n_samples
            samples = sampler.random(n=self.n_samples)

            # Scale samples to the parameter ranges
            scaled_samples = qmc.scale(samples, [r[0] for r in ranges], [r[1] for r in ranges])

            # Convert the samples to a list of dictionaries, each representing a combination
            precision = 3  # Set to 2 or 3 as needed
            all_combinations = scaled_samples# [{param: round(value, precision) for param, value in zip(parameters, sample)} for sample in scaled_samples]
            n_lut_build = len(all_combinations)
            #pdb.set_trace()
        # ASCII banner
        print("*" * 60)
        print("*" + " " * 58 + "*")
        print("*   Starting MODTRAN simulation with the following settings:  *")
        print("*" + " " * 58 + "*")
        print("*" * 60)
        
        print("\nWe are assuming each simulation takes 1.5 minutes, using the 05_2013 band model.")
        print("We are assuming we will run with 15 nodes on the EMIT cluster.\n")

        print('Num LUTs to build: {}'.format(n_lut_build))
        print('Expected MODTRAN runtime: {} minutes'.format(n_lut_build*1))
        print('Expected MODTRAN runtime: {} hrs'.format(n_lut_build*1.5/60))
        print('Expected MODTRAN runtime: {} days'.format(n_lut_build*1.5/60/24))
        print('Expected MODTRAN runtime per (all core) node: {} hours'.format(n_lut_build*1.5/60/15))
        print('Expected MODTRAN runtime per (all core) node: {} days'.format(n_lut_build*1.5/60/24/15))
        #pdb.set_trace()

        id = 0
        files_per_directory = 5000
        #self.output_dir+batch_n

        for combination in all_combinations:
            id += 1
            # Determine the directory based on the id
            dir_index = (id - 1) // files_per_directory
            batch_n = 'batch_%05d' % dir_index
            directory_path = os.path.join(self.output_dir, batch_n)

            # Create the directory if it doesn't exist
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)



            # update mod_input with the current combination of parameter values
            

            if hasattr(self, 'max_per_gndalt'):
                # we need to translate between the relative and the absolute
                current_gndalt = parameters.index('GNDALT')
                current_relative_h2o = combination[parameters.index('H2OSTR')]
                # now we must find the entru in max_per_gndalt
                arr = self.max_per_gndalt
                max_water = arr[arr[:, 0] == current_gndalt, 1]
                absolute_water = current_relative_h2o * max_water
                if absolute_water==0:
                    absolute_water = 0.01
                
                combination_list = list(combination)

                combination_list[parameters.index('H2OSTR')] = absolute_water
                combination = combination_list
                
            #pdb.set_trace()
            mod_input.update({param: float(value) for param, value in zip(parameters, combination)})
            # The above will fail if mod_input doesn't have all the keys initiated.
            #for param, value in zip(parameters, combination):
            #    mod_input[param] = float(value)

            #pdb.set_trace()
            outf = '_'.join(['%s-%6.4f' % (param, mod_input[param]) for param in parameters])
            #outf = outf + '_ID-%06d' % id
            if hasattr(self, 'max_per_gndalt'):
                outf = outf + '_CRH-%06.4f' % current_relative_h2o
            mod_input['NAME'] = outf
            #pdb.set_trace()

            if hasattr(self, 'allow_rh_ht_100'):
                allow_rh_above_100 = self.allow_rh_hr_100
            else:
                allow_rh_above_100 = False
            # This is the first output template and whatever we do here goes to all cases
            lut_point_json = get_modtran_template_one_part(mod_input, allow_rh_above_100)
            

            if False:#self.use_aerosol_model is True:
                lut_point_json['MODTRAN'][0]['MODTRANINPUT']['AEROSOLS'] = self.aerosol_tmpl
                #pdb.set_trace()
                AOD_param_idx = int(parameters.index('AOT550'))
                lut_point_json['MODTRAN'][0]['MODTRANINPUT']['AEROSOLS']['VIS'] = -combination[AOD_param_idx]

            #pdb.set_trace()

            # I'm going to put the angstrom exponent here for now
            #pdb.set_trace()
            if 'ASTMX' in mod_input and mod_input['ASTMX'] is not None:
                # we are changing the angstrom exponent for both the bountry layer and the toposphere
                # I tried with the other option too and it doesn't make a difference, but we still have to put a value here
                lut_point_json['MODTRAN'][0]['MODTRANINPUT']['AEROSOLS']['CDASTM'] = "T" 
                lut_point_json['MODTRAN'][0]['MODTRANINPUT']['AEROSOLS']['ASTMX'] = mod_input['ASTMX']


            param = lut_point_json["MODTRAN"]
            #pdb.set_trace()
            if self.multipart is True and self.multiband is False:
                # here is an explantion from Lex about what is the meaning of the reflectance values:
                # Remember, the atmospheric correction equation (ACE) is only valid monochromatically.
                #  With the 3 albedo approach, one is performing a bet fit to the ACE. 
                # It helps to include a calculation whose reflectance is close to the actual. 
                # If you know you do not have bright objects in your scene, 
                # you might even choose to use [0.0, 0.1, 0.5].

                test_rfls = [0, 0.1, 0.5]
                param = []
                jsontmpl = lut_point_json["MODTRAN"][0]
                # Here we copy the original config and just change the surface reflectance
                jsontmpl["MODTRANINPUT"]["CASE"] = 0
                jsontmpl["MODTRANINPUT"]["SURFACE"]["SURREF"] = test_rfls[0]#self.test_rfls[0]
                param.append(jsontmpl)
                lut_point_json1 = deepcopy(jsontmpl)
                lut_point_json1["MODTRANINPUT"]["CASE"] = 1
                lut_point_json1["MODTRANINPUT"]["SURFACE"]["SURREF"] = test_rfls[1]
                param.append(lut_point_json1)
                lut_point_json2 = deepcopy(jsontmpl)
                lut_point_json2["MODTRANINPUT"]["CASE"] = 2
                lut_point_json2["MODTRANINPUT"]["SURFACE"]["SURREF"] = test_rfls[2]
                param.append(lut_point_json2)
            
            elif self.multipart is True and self.multiband is True:
                # we need overall 3 tiplet cases here 3x3=9:
                # first triplet is the three albedo for VIS with 5cm-1
                # second is NIR-SWIR1 with 1cm-1
                # Finally we run the p5
                #pdb.set_trace()
                # note that we are writing all the cases into the same csv file!
                # It would just append, not rewrite!
                test_rfls = [0, 0.1, 0.5]
                param = []
                jsontmpl_11 = deepcopy(lut_point_json["MODTRAN"][0])
                # Only the first case will have a tp6 with for it. I don't know what happens otherwise
                jsontmpl_11['MODTRANINPUT']['FILEOPTIONS']['NOFILE'] = "FC_TAPE6ONLY"
                jsontmpl_11['MODTRANINPUT']['FILEOPTIONS']['NOPRNT'] = 2
                jsontmpl_11['MODTRANINPUT']['FILEOPTIONS'].pop('CKPRNT')
                name = jsontmpl_11['MODTRANINPUT']['NAME']
                csvfname = name+'.csv'
                

                jsontmpl_11['MODTRANINPUT']['SPECTRAL']['BMNAME'] = "05_2013"
                jsontmpl_11['MODTRANINPUT']['SPECTRAL']['V1'] = 340
                jsontmpl_11['MODTRANINPUT']['SPECTRAL']['V2'] = 817
                jsontmpl_11["MODTRANINPUT"]["CASE"] = 0
                jsontmpl_11["MODTRANINPUT"]["SURFACE"]["SURREF"] = test_rfls[0]
                jsontmpl_11['MODTRANINPUT']['FILEOPTIONS']['CSVPRNT'] = csvfname
                param.append(jsontmpl_11)

                jsontmpl_12 = deepcopy(jsontmpl_11)
                jsontmpl_12["MODTRANINPUT"]["CASE"] = 1
                jsontmpl_12["MODTRANINPUT"]["SURFACE"]["SURREF"] = test_rfls[1]
                jsontmpl_12['MODTRANINPUT']['FILEOPTIONS']['CSVPRNT'] = csvfname
                param.append(jsontmpl_12)

                jsontmpl_13 = deepcopy(jsontmpl_11)
                jsontmpl_13["MODTRANINPUT"]["CASE"] = 2
                jsontmpl_13["MODTRANINPUT"]["SURFACE"]["SURREF"] = test_rfls[2]
                jsontmpl_13['MODTRANINPUT']['FILEOPTIONS']['CSVPRNT'] = csvfname
                param.append(jsontmpl_13)

                jsontmpl_21 = deepcopy(lut_point_json["MODTRAN"][0])
                jsontmpl_21['MODTRANINPUT']['FILEOPTIONS'].pop('CKPRNT')
                jsontmpl_21['MODTRANINPUT']['FILEOPTIONS']['NOFILE'] = "FC_NOFILES"
                jsontmpl_21['MODTRANINPUT']['FILEOPTIONS']['NOPRNT'] = 2
                name = jsontmpl_21['MODTRANINPUT']['NAME']

                jsontmpl_21['MODTRANINPUT']['CASE'] = 3
                jsontmpl_21['MODTRANINPUT']['SPECTRAL']['BMNAME'] = "01_2013"
                jsontmpl_21['MODTRANINPUT']['SPECTRAL']['V1'] = 817
                jsontmpl_21['MODTRANINPUT']['SPECTRAL']['V2'] = 1826
                jsontmpl_21["MODTRANINPUT"]["SURFACE"]["SURREF"] = test_rfls[0]
                jsontmpl_21['MODTRANINPUT']['FILEOPTIONS']['CSVPRNT'] = csvfname
                param.append(jsontmpl_21)

                jsontmpl_22 = deepcopy(jsontmpl_21)
                jsontmpl_22['MODTRANINPUT']['CASE'] = 4
                jsontmpl_22["MODTRANINPUT"]["SURFACE"]["SURREF"] = test_rfls[1]
                jsontmpl_22['MODTRANINPUT']['FILEOPTIONS']['CSVPRNT'] = csvfname
                param.append(jsontmpl_22)

                jsontmpl_23 = deepcopy(jsontmpl_21)
                jsontmpl_23['MODTRANINPUT']['CASE'] = 5
                jsontmpl_23["MODTRANINPUT"]["SURFACE"]["SURREF"] = test_rfls[2]
                jsontmpl_23['MODTRANINPUT']['FILEOPTIONS']['CSVPRNT'] = csvfname
                param.append(jsontmpl_23)

                jsontmpl_31 = deepcopy(lut_point_json["MODTRAN"][0])
                jsontmpl_31['MODTRANINPUT']['FILEOPTIONS'].pop('CKPRNT')
                jsontmpl_31['MODTRANINPUT']['FILEOPTIONS']['NOFILE'] = "FC_NOFILES"
                jsontmpl_31['MODTRANINPUT']['FILEOPTIONS']['NOPRNT'] = 2
                name = jsontmpl_31['MODTRANINPUT']['NAME']

                jsontmpl_31['MODTRANINPUT']['CASE'] = 6
                jsontmpl_31['MODTRANINPUT']['SPECTRAL']['BMNAME'] = "p1_2013"
                jsontmpl_31['MODTRANINPUT']['SPECTRAL']['V1'] = 1826
                jsontmpl_31['MODTRANINPUT']['SPECTRAL']['V2'] = 2520
                jsontmpl_31["MODTRANINPUT"]["SURFACE"]["SURREF"] = test_rfls[0]
                jsontmpl_31['MODTRANINPUT']['FILEOPTIONS']['CSVPRNT'] = csvfname
                param.append(jsontmpl_31)

                jsontmpl_32 = deepcopy(jsontmpl_31)
                jsontmpl_32['MODTRANINPUT']['CASE'] = 7
                jsontmpl_32["MODTRANINPUT"]["SURFACE"]["SURREF"] = test_rfls[1]
                jsontmpl_32['MODTRANINPUT']['FILEOPTIONS']['CSVPRNT'] = csvfname
                param.append(jsontmpl_32)

                jsontmpl_33 = deepcopy(jsontmpl_31)
                jsontmpl_33['MODTRANINPUT']['CASE'] = 8
                jsontmpl_33["MODTRANINPUT"]["SURFACE"]["SURREF"] = test_rfls[2]
                jsontmpl_33['MODTRANINPUT']['FILEOPTIONS']['CSVPRNT'] = csvfname
                param.append(jsontmpl_33)





            #output_file = self.output_dir+batch_n+'/'+outf+'.json'
            #pdb.set_trace()
            output_file = os.path.join(directory_path, outf + '.json')
            with open(output_file, 'w') as fout:
                fout.write(json.dumps({"MODTRAN": param}, cls=SerialEncoder, indent=4, sort_keys=True))
            #pdb.set_trace()



    def run_modtran(self, ncores):
        # This is not used from this code abymore, but I'll keep it here for now anyways until we redo the code
        modtran_exe = os.getenv('MODTRAN_EXE')
        #modtran_exe = '/beegfs/store/shared/MODTRAN6/MODTRAN6.0.0/bin/linux/mod6c_cons'
        if modtran_exe is None:
            print("Please define an env variable for modtran exe or hard code it in here")
            modtran_exe = '/opt/ESS/MODTRAN-6.0/bin/linux/mod6c_cons'
        
        runm(self.lutdir, ncores, modtran_exe)







    def read_chns_ray(self):
        # not sure if this function is needed, we can clean it up
        results = self.get_obj_res()


    def get_obj_res(self):
        # Get the parent directory
        parent_dir = os.path.dirname(self.lutdir)
        #pdb.set_trace()
        # Replace 'output' with 'new_output' in the directory path
        if self.batch_dir is not None:
            lutdir = self.batch_dir
        else:
            lutdir = os.path.join(parent_dir, 'output') # look here for "new_output" if doing rereading the bounds
        
        
        
        # Initialize Ray for parallel processing
        rayargs = {'local_mode': False, 'num_cpus': mp.cpu_count()}
        #rayargs = {'local_mode': True, 'num_cpus': 1}
        ray.init(**rayargs)
        
        # Put the current object into the Ray object store
        objid = ray.put(self)
        
        # Initialize a list to hold the results
        results = []
        
        # Create a list of jobs
        #pdb.set_trace()
        if self.multiband is True:
            # We have to add a check to see if the value of water or aerosols changed!
            # Then we have to maybe get the grid from the config and use that instead of just reading whatever
            # files are there.......
            # if multi band we are reading the tp7 .csv file, not channel

            if self.sampling == 'regular' or self.sampling == 'relative_water':
                mod_input = {
                'atmosphere_type': 'ATM_TROPICAL',
                'alt': float(self.input_dict['H1'][0]),
                'aerosol_model': 'AER_RURAL',
                'filter_file': self.filter_file,
                'TSZ': float(self.input_dict['TSZ'][0]),
                'RAA': float(self.default_angle),
                'band_model': self.band_model
                }


                
                param_values = [self.input_dict[param] for param in self.state_vecs]
                all_combinations = list(product(*param_values))
                tp7_csv_files_should_be = []
                parameters = self.state_vecs
                for combination in all_combinations:


                    if hasattr(self, 'max_per_gndalt'):
                        # we need to translate between the relative and the absolute
                        current_gndalt = parameters.index('GNDALT')
                        current_relative_h2o = combination[parameters.index('H2OSTR')]
                        # now we must find the entru in max_per_gndalt
                        arr = self.max_per_gndalt
                        max_water = arr[arr[:, 0] == current_gndalt, 1]
                        absolute_water = current_relative_h2o * max_water
                        if absolute_water==0:
                            absolute_water = 0.01
                        
                        combination_list = list(combination)

                        combination_list[parameters.index('H2OSTR')] = absolute_water
                        combination = combination_list
                        
                    #pdb.set_trace()
                    mod_input.update({param: float(value) for param, value in zip(parameters, combination)})
                    # The above will fail if mod_input doesn't have all the keys initiated.
                    #for param, value in zip(parameters, combination):
                    #    mod_input[param] = float(value)

                    #pdb.set_trace()
                    outf = '_'.join(['%s-%6.4f' % (param, mod_input[param]) for param in parameters])
                    #outf = outf + '_ID-%06d' % id
                    if hasattr(self, 'max_per_gndalt'):
                        outf = outf + '_CRH-%06.4f' % current_relative_h2o

                    

                    tp7_csv_files_should_be.append(outf+'.csv')
                #pdb.set_trace()
                
                tp7_csv_files_in_dir = glob.glob(os.path.join(lutdir, '*[0-9].csv'))
                tp7_csv_files_in_dir = [os.path.basename(filepath) for filepath in tp7_csv_files_in_dir]

                #pdb.set_trace()
                missing_files = [file for file in tp7_csv_files_should_be if file not in tp7_csv_files_in_dir]
                if len(missing_files) == 0:
                    print("No missing csv files")
                    RT_files = tp7_csv_files_should_be
                #import pdb; pdb.set_trace()
                #RT_files = [file for file in tp7_csv_files if not file.endswith(('chan.csv', 'flux.csv', 'scan.csv'))]
            else:
                tp7_csv_files_in_dir = glob.glob(os.path.join(lutdir, '*[0-9].csv'))
                tp7_csv_files_in_dir = [os.path.basename(filepath) for filepath in tp7_csv_files_in_dir]
                RT_files = tp7_csv_files_in_dir
        
        else:
            # Get the list of .chn files in the directory
            RT_files = glob.glob(os.path.join(lutdir, '*.chn'))
        
        #pdb.set_trace()
        jobs = [read_data_piece.remote(ind, os.path.join(lutdir, chn_file), objid, self.multipart) for ind, chn_file in enumerate(RT_files)]
        #num_samples_to_process = 5000
        #jobs = [read_data_piece.remote(ind, chn_file, objid, self.multipart) for ind, chn_file in enumerate(RT_files[:num_samples_to_process])]
        # Get the results of the jobs with a progress bar
        failed_filenames_file = '/scratch/carmon/modtran_luts/projects/EMIT_Reprocessing/failed_files.txt'
        for _ in tqdm(range(len(jobs)), desc="Processing files"):
            done_id, _ = ray.wait(jobs)
            ind, res = ray.get(done_id[0])
            jobs.remove(done_id[0])
            
            if res is not None:
                results.append(res)
            else:
                pass
                # Write the failed filename to the text file
                #with open(failed_filenames_file, 'a') as f:
                #    failed_filename = RT_files[ind]  # Assuming RT_files[ind] gives the filename
                #    f.write(f"{failed_filename}\n")  # Append the failed filename to the file

        
        # Save the results to self.universal_lut
        #pdb.set_trace()
        self.universal_lut = results

    def get_obj_res2(self):
        # get chn file names
        # if we are here it means we are reading from the dir
        # currently we have a new dir new_output with symlinks and correct values in the file names
        #pdb.set_trace()
        parent_dir = os.path.dirname(self.lutdir)
        lutdir = os.path.join(parent_dir, 'new_output')
        chn_files = glob.glob(lutdir+'/*.chn', recursive=True)
        self.files = chn_files
        # We don't want the VectorInterpolator, but rather the raw inputs
        import multiprocessing as mp
        # set up ray for parallel processing
        rayargs = {'local_mode':False, 'num_cpus':mp.cpu_count()}
        #rayargs = {'local_mode':True, 'num_cpus':1}
        ray.init(**rayargs)
        #param, response = load_chn(chn_files[0])

        #### standard ray stuff
        # make a jobs list with the remote function calls
        results = []
        objid = ray.put(self)
        jobs = []


        for ind, chn_file in enumerate(chn_files):
            jobs.append(read_data_piece.remote(ind, chn_file, objid, self.multipart))
        rreturn = [ray.get(jid) for jid in jobs]


        
        # go over each processed file and append to results
        for ind, res in rreturn:
            if res is not None:
                try:
                    #pdb.set_trace()
                    results.append(res)
                except:
                    print(f'failed on file: {chn_file}')
                    #results[ind,:] = np.nan
            else:
                print(f'failed on file: {chn_file}')
                #results[ind,:] = np.nan
        # save list to self.unicersal_lut
        self.universal_lut = results
        

    
    def saveToHDF5(self, lut_cnf):
        # this is called from read_and_save function in apply_lut

        try:
            if self.batch_dir is not None:
                dir_path = self.batch_dir
            else:
                dir_path = self.output_dir  # set the directory path to search (use "." for the current directory)
            extension = ".json"  # set the file extension to search for

            # list all files in the directory
            files = os.listdir(dir_path)

            # filter the list to only include files with the desired extension
            txt_files = [file for file in files if file.endswith(extension)]

            # load first json
            mod_cnfg = json.load(open(os.path.join(dir_path, txt_files[0]), 'rb'))
        
        except:
            mod_cnfg = None

        # we use this SaveToHDF5 function now, which is in format_hdf5.py
        # we should definitly change the name because it's confusing.
        #pdb.set_trace()
        if self.multiband is True:
            SaveToHDF5_multimodel(self.universal_lut, lut_cnf, mod_cnfg)
        else:
            SaveToHDF5(self.universal_lut, lut_cnf, mod_cnfg)
        # Basically this is the end of this codebase. We do have 'define sample space' but that is not directly related to here



@ray.remote
def read_data_piece(ind, rt_file, obj, multipart):
    max_retry = 10
    retry_delay = 1  # Seconds to wait between retries
    failed_files_log = '/scratch/carmon/modtran_luts/projects/EMIT_Reprocessing/failed_files.txt'
    for attempt in range(max_retry):
        try:
            point = parse_fname(rt_file)
            if rt_file.endswith(".chn"):
                mod_out = load_chn_single(rt_file, multipart)
            elif rt_file.endswith(".csv"):
                mod_out = process_tp7(rt_file)
            return ind, (point, mod_out)  # Successful read, return result

        except FileNotFoundError as e:
            message = f"Attempt {attempt + 1}: Failed to read {rt_file}: {type(e).__name__} at line {e.__traceback__.tb_lineno} - {e}"
            print(message)
            time.sleep(retry_delay)  # Wait before retrying

        except Exception as e:  # Other exceptions
            message = f"Unexpected error: {type(e).__name__} at line {e.__traceback__.tb_lineno} - {e}"
            print(message)
            print("Bad file is" + rt_file)
            with open(failed_files_log, 'a') as log_file:  # Open log file in append mode
                log_file.write(f"{rt_file + ':' + message}\n")  
            return ind, (None, None)  # Return indicating failure due to unexpected error

    # Max retries reached, return failure
    print(f"Max retries reached. Failed to read {rt_file}")
    return ind, (None, None)



def parse_fname(fname_string):
    # Create a dictionary to hold the parameter values
    point1 = {}
    #pdb.set_trace()
    # Remove the '.chn' extension and get the last part of the filename
    filename, file_extension = os.path.splitext(fname_string)
    gridname = filename.split('/')[-1]
    #gridname = fname_string.strip('.chn').split('/')[-1]

    # Extract parameters from the filename
    # for people who are curious about this re expression:
    # Regular Expression Explanation:

    # Overall, this regular expression matches a parameter name consisting of one or more characters other than underscore and hyphen, 
    # followed by a hyphen, followed by a possibly negative decimal number. The parameter name and value are captured as separate groups 
    # and can be extracted by the `re.findall` function.


    parameters = re.findall(r"([^_-]+)-(-?\d*\.?\d*)", gridname)

    # Add parameter values to the dictionary
    for name, val in parameters:
        point1[name] = float(val)

    return point1


def parse_fname2(fname_string):

    point1 = {}

    gridname = fname_string.strip('.chn').split('/')[-1].split('_')


    for lut_dim in gridname:
        try:
            name=lut_dim.split('-')[0]
            val = lut_dim.split('-')[1]
            point1[str(name)] = float(val)
            #if name=='SZA':
            #    coszen = np.round(np.cos(np.deg2rad(float(val))), 3)
        except:
            pass

    return point1

def extract_water_vapor_data_from_tp6(directory):
    """
    Extract ground altitude and adjusted water vapor values from .tp6 files in the given directory.
    """
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".tp6"):
            gnd_alt = extract_data_from_filename(filename)
            with open(os.path.join(directory, filename), 'r') as file:
                contents = file.read()
                h2o_str = extract_water_vapor_warning(contents)
                if gnd_alt is not None and h2o_str is not None:
                    data.append([gnd_alt, h2o_str])
    return np.array(data)

def extract_data_from_filename(filename):
    """
    Extract ground altitude from the filename.
    """
    match = re.search(r"GNDALT-(-?[\d.]+)", filename)
    if match:
        return float(match.group(1))
    else:
        return None

def extract_water_vapor_warning(contents):
    """
    Extract the adjusted water vapor value from the .tp6 file contents.
    """
    match = re.findall(r"Warning from routine SCLCOL:.*Input water column,.*is (above maximum allowed|below minimum allowed).*maximum, (.*?) gm/cm2", contents, re.DOTALL)
    if match:
        return float(match[0][1].strip())
    else:
        return None
    
class SerialEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return super(SerialEncoder, self).default(obj)
