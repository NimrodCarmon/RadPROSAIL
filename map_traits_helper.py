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
import numpy as np
from scipy.optimize import least_squares
#from prosail2 import Prosail
import matplotlib.pyplot as plt
#import numba
from helper2 import dataSpec_P5B
import csv
import matplotlib.pyplot as plt
from pyrsr import RelativeSpectralResponse
import pdb
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def resample_by_instrument_name(dbase, inst_name = 'ls8'):
    infile = open(dbase, 'r')
    lines=infile.readlines()
    headers=[] # Row headers, describe what these data sets mean.  Not sure they're of much utility in the code.
    data=[]
    for line in lines:
        line=line.rstrip()
        vals=line.split(',')
        headers.append(vals[0])
        datavals=vals[1:]
        datavals=[float(v) for v in datavals]
        data.append(datavals)
    infile.close()
    data = np.asarray(data)

    if inst_name == 'ls8':
        
        RSR = RelativeSpectralResponse(satellite="Landsat-8", sensor="OLI_TIRS")
        wvl = RSR.rsrs_wvl # wvl are in nm, starting at 427, ending at 1400, band every 1 nm.
        srf = RSR.rsrs

    lastband = np.argmax(wvl>2505)
    wvl = wvl[0:lastband]
    srf = dict([(key,val[0:lastband]) for key, val in srf.items()])

    
    # we should couple the min max of the two grids
    intersection = np.intersect1d(wvl, data[0])
    data_indices = np.where(np.isin(data[0], intersection))[0]
    wvl_indices = np.where(np.isin(wvl, intersection))[0]
    #pdb.set_trace()
    transformation_matrix = []
    wvl_subsetted = wvl[wvl_indices]

    target_wvs = []
    for band in range(1, 8):
        response = srf[str(int(band))]
        subsetted_response = np.array(response)[wvl_indices]
        subsetted_response /= np.sum(subsetted_response)
        transformation_matrix.append(subsetted_response)
        maxval = np.argmax(subsetted_response)
        target_wvs.append(wvl_subsetted[maxval])

    # we need to transform data[1:] because the first is the wavelength
    transformation_matrix = np.array(transformation_matrix)
    resampled = []
    for idx, spec in enumerate(data):
        if idx == 0: pass
        else:
            resampled.append(np.dot(transformation_matrix, spec[data_indices]).tolist())


    # the wvl and srf coming from pyrsr are already in 1nm intervals, same as the prosail lib
    # So we just need to create a transformation matrix from the srf


    outdata = [target_wvs]+ resampled
    
    for i in range(len(outdata)):
        outdata[i].insert(0, headers[i])
    


    outfname = dbase.split('.')[0]+'_resampled.csv'
    with open(outfname, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(outdata)




def resample_prosail_database(dbase, wvs_target):
    # we take the original database and rewrite it in a resampled format
    infile = open(dbase, 'r')
    lines=infile.readlines()
    headers=[] # Row headers, describe what these data sets mean.  Not sure they're of much utility in the code.
    data=[]
    for line in lines:
        line=line.rstrip()
        vals=line.split(',')
        headers.append(vals[0])
        datavals=vals[1:]
        datavals=[float(v) for v in datavals]
        data.append(datavals)
    infile.close()


    wvs_base = np.array(data[0])
    rr = resample(wvs_base, wvs_target['wvs'], wvs_target['fwhm'])
    resampler = rr.resample_data

    resampled = [list(resampler(item)) for item in data[1:]]
    # for the future person that will read this and wonder why I use [:]
    # the reason is that python references the list and not actually saves it new
    # so it would change the original ds.bands.centers object if we didn't do that.

    outdata = [wvs_target['wvs'][:]]+ resampled
    
    for i in range(len(outdata)):
        outdata[i].insert(0, headers[i])
    

    #import pdb; pdb.set_trace()
    outfname = dbase.split('.')[0]+'_resampled.csv'
    with open(outfname, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(outdata)
    #return headers, data



def estimate_traits(rfl, obs, band_names, wvs_targett, prosail):
    # get wavelengths
    # first thing we do is if NDVI is smaller than 0.2, we don't calculate and return nan
    # when I built the files I've made a mistake, and have the fwhm in nanometers and wavelengths in microns
    #import pdb; pdb.set_trace()
    if wvs_targett['wvs'][0] < 300:
        wvs_targett['wvs'] = [wv * 1000 for wv in wvs_targett['wvs']]
    if wvs_targett['fwhm'][0] <= 1:
        wvs_targett['fwhm'] = [f * 1000 for f in wvs_targett['fwhm']]

    veg_or_not = calc_ndvi(rfl, wvs_targett['wvs'], threshold = 0.2)
    if not veg_or_not:
        nan_list = [np.nan] * 11 * 2
        return nan_list
    else:
        try:
            #pdb.set_trace()
            p = prosail

            obj = EstimateTraits(wvs_targett['wvs'], p)
            x_hat = obj.estimate(rfl, obs, band_names)
            return x_hat
        except:
            #print("Something with the inversion went wrong")
            nan_list = [np.nan] * 11 * 2
            return nan_list

def calc_ndvi(rfl, wvs, threshold):
    red_band_nm = 650
    nir_band_nm = 860

    # Find the nearest wavelengths in the spectrum data
    red_band_index = min(range(len(wvs)), key=lambda i: abs(wvs[i] - red_band_nm))
    nir_band_index = min(range(len(wvs)), key=lambda i: abs(wvs[i] - nir_band_nm))

    # Get the data from the spectrum for the red and NIR bands
    red_band_data = rfl[red_band_index]
    nir_band_data = rfl[nir_band_index]

    # Calculate NDVI
    ndvi = (nir_band_data - red_band_data) / (nir_band_data + red_band_data)

    # Check if NDVI is less than 0.2
    if ndvi < threshold:
        return False
    else:
        # Perform some other operation if NDVI is greater than or equal to 0.2
        # Replace the following line with the desired action
        return True

class EstimateTraits():
    def __init__(self, wvs, prosail):
        lim_values = {'N': [0, 2], 'Cab': [0, 80], 'Car': [0, 20], 'Cbrown': [0, 1], \
            'Cw': [0, 0.2], 'Cm': [0.003, 0.05], 'LAI': [0, 10], 'psoil': [0, 1], \
                'hspot': [0, 1], 'LIDFa': [-1, 1], 'LIDFb': [-1, 1]}
        
        #self.config = {
        #    'max_nfev': 10,
        #    'method': 'trf',
        #    'bounds': ([value[0] for value in lim_values.values()], [value[1] for value in lim_values.values()])
        #}

        self.config = {
            'method': 'trf', # Trust Region Reflective algorithm
            #'jac': '2-point', # Use a two-point finite difference approximation for the Jacobian
            'max_nfev': 30, # Maximum number of function evaluations
            'gtol': 1e-6, # The optimization stops when the maximum of the absolute values of the gradient is below this threshold
            #'x_scale': 'jac', # The variables will be rescaled internally for the algorithm to work well, based on the Jacobian at 'x0'
            'loss': 'linear', # The loss function rho(y) = y. Gives a standard least-squares problem.
            'verbose': 0, # The verbosity level of the function. If it is zero, the function does not output anything. If it is 1, the final result will be printed. If it is 2, progress during iterations will be printed.
            'bounds': ([value[0] for value in lim_values.values()], [value[1] for value in lim_values.values()])
        }

        #parameter_names = ["N", "Cab", "Car", "Cbrown", "Cw", "Cm", "LAI", "psoil", "hspot"]

        self.wvs = wvs
        self.p = prosail
        # maybe here we should set up the resampling too
    
    def project_rfl(self,x_hat):
        rho_hat = self.p.run(self.get_params_in_config(x_hat))
        return rho_hat

    def estimate(self, rfl, obs, band_names):
        
        tts = obs[band_names.index('To-sun zenith (0 to 90 degrees from zenith)')]
        tto = obs[band_names.index('To-sensor zenith (0 to 90 degrees from zenith)')]

        observation_zenith = obs[band_names.index('To-sensor azimuth (0 to 360 degrees CW from N)')]
        solar_zenith = obs[band_names.index('To-sun azimuth (0 to 360 degrees CW from N)')]
        # relative azimuth angle
        psi = np.abs(observation_zenith - solar_zenith)

        #self.p = Prosail()
        # we have 9 free parameters
        self.cnfg_tmpl = {'tts': tts, 'tto':tto, 'psi':psi}#, 'LIDF': (-0.35, -0.15)}

        # try cost function
        x0 = [0.5, 40, 8, 0.01, 0.01, 0.009, 5, 0.5, 0.5, -0.35, -0.15]
        #pdb.set_trace()

        result = least_squares(self.cost_function, x0, jac=self.custom_jacobian, **self.config, args = (rfl,))
        x_hat = result.x


        '''rfl_hat = self.project_rfl(x_hat)
        #plt.plot(self.wvs, rfl, label='observed')
        #plt.plot(self.wvs, rfl_hat, label='estimate')
        #plt.legend()
        #plt.savefig('img.jpg')
        #pdb.set_trace()

        param_se = self.estimate_uncertainty(result.jac, rfl, rfl_hat, x_hat)
        #pdb.set_trace()
        # adjust output for LIDF
        LIDFa_original = x_hat[-2]
        LIDFb_original = x_hat[-1]

        LIDFa_adjusted = LIDFa_original * np.sqrt(1-LIDFb_original**2)
        LIDFb_adjusted = LIDFb_original * np.sqrt(1-LIDFa_original**2)

        x_hat[-2] = LIDFa_adjusted
        x_hat[-1] = LIDFb_adjusted

        # calculate the gradient of the transformation functions with respect to the original parameters
        grad_a = np.sqrt(1 - LIDFb_original**2) - LIDFa_original * LIDFb_original / np.sqrt(1 - LIDFb_original**2)
        grad_b = np.sqrt(1 - LIDFa_original**2) - LIDFb_original * LIDFa_original / np.sqrt(1 - LIDFa_original**2)

        # apply the delta method
        se_a_adjusted = np.abs(grad_a * param_se[-2])
        se_b_adjusted = np.abs(grad_b * param_se[-1])

        param_se[-2] = se_a_adjusted
        param_se[-1] = se_b_adjusted'''

        #return np.hstack((x_hat, param_se))
        return np.hstack((x_hat, np.sqrt(abs(x_hat))))




    def cost_function(self, params, y_data):
        # A dirty fix to force a leaf distribution to be spherical 

        params[-2] = -0.35
        params[-1] = -0.15
        config = self.get_params_in_config(params)
        y_hat = self.p.run(config)

        if np.isnan(y_hat).any():
            # Handle NaN values as needed; for example, you could return a large cost
            return self.subset_spectrum(np.full_like(y_data, 1e6))  # Replace with an appropriate value

        cost = y_data - y_hat
        cost = self.subset_spectrum(cost)

        return cost




    def custom_jacobian(self, params, *args):
        # the prob here is that the cost returns a subsetted response but jac is not subsetted
        epsilon = 0.01  # Customize this value as needed
        n_params = len(params)
        y_data = args[0]

        # Get subsetted size from the initial call to cost_function
        initial_res = self.cost_function(params, y_data)
        subsetted_size = len(initial_res)

        # Initialize the Jacobian matrix with the subsetted size
        jac = np.zeros((subsetted_size, n_params))

        for i in range(n_params):

            params1 = params.copy()
            params2 = params.copy()
            params1[i] -= epsilon
            params2[i] += epsilon

            res1 = self.cost_function(params1, y_data)
            res2 = self.cost_function(params2, y_data)

            jac[:, i] = (res2 - res1) / (2 * epsilon)

            # Debugging: Check if there are any NaNs or Infs
            if np.isnan(jac[:, i]).any() or np.isinf(jac[:, i]).any():
                print(f"NaN or Inf detected in Jacobian at parameter index {i}.")
                print("params1:", params1)
                print("params2:", params2)
                print("res1:", res1)
                print("res2:", res2)

        return jac



    def subset_spectrum(self, spectrum):
        windows = [
            [350.0, 1360.0],
            [1410.0, 1800.0],
            [1970.0, 2500.0]
        ]

        subset = []
        for window in windows:
            window_start, window_end = window
            window_indices = [i for i, wavelength in enumerate(self.wvs) if window_start <= wavelength <= window_end]
            subset.extend(spectrum[window_indices])

        return np.squeeze(np.array(subset))

    def get_params_in_config(self, params_values):


        cnfg = self.cnfg_tmpl
        parameter_names = ["N", "Cab", "Car", "Cbrown", "Cw", "Cm", "LAI", "psoil", "hspot", 'LIDFa', 'LIDFb']
        
        for name, value in zip(parameter_names, params_values):
            cnfg[name] = value
        
        # LIDFa and LIDFb are a special case in terms of the state vector
        # for the solver they appear within the range [-1, 1]
        # but PROSAIL must have them with the constraint that |LIDFa| + |LIDFb| <= 1
        # So we will do the following trick:

        LIDFa_original = cnfg['LIDFa']
        LIDFb_original = cnfg['LIDFb']
        
        LIDFa_adjusted = LIDFa_original * np.sqrt(1-LIDFb_original**2)
        LIDFb_adjusted = LIDFb_original * np.sqrt(1-LIDFa_original**2)

        cnfg['LIDFa'] = LIDFa_adjusted
        cnfg['LIDFb'] = LIDFb_adjusted
        
        return cnfg

    def estimate_uncertainty(self, jac, y_observed, y_predicted, x_hat):
        """
        Estimate the standard error of the parameters using the Jacobian matrix 
        and the residuals of the model.

        Parameters
        ----------
        jac: array-like
            The Jacobian matrix of the function.
        y_observed: array-like
            The observed outcomes.
        y_predicted: array-like
            The outcomes predicted by the model.
        x_hat: array-like
            The estimated parameters of the model.

        Returns
        -------
        param_sd: array-like
            The standard error of the parameters.
        """

        # Calculate the residuals, i.e., the difference between the observed and predicted outcomes.
        residuals = y_observed - y_predicted
        # the cost function subsets the residuals on the valid spectral windows, so we do it here too
        residuals = self.subset_spectrum(residuals)
        
        # Calculate the sum of squared residuals (sse).
        # This is a measure of the discrepancy between the data and the model, i.e., the goodness of fit.
        sse = np.sum(residuals**2)
        
        # Calculate the degrees of freedom (df).
        # This is the number of data points minus the number of parameters estimated by the model.
        df = len(residuals) - len(x_hat)
        
        # Calculate the covariance matrix of the parameters.
        # This is done by inverting the dot product of the Jacobian matrix and its transpose,
        # and then multiplying by the mean square error (mse), which is the sse divided by the df.
        # The mse is an estimate of the variance of the residuals.
        cov_matrix = np.linalg.inv(np.dot(jac.T, jac)) * (sse / df)
        
        # Calculate the standard error of the parameters.
        # This is done by taking the square root of the diagonal of the covariance matrix.
        # The diagonal of the covariance matrix represents the variance of each parameter, 
        # and the square root of the variance gives the standard deviation, i.e., the standard error.
        param_sd = np.sqrt(np.diag(cov_matrix))

        return param_sd
    
    def get_out():
        import os
        os._exit(0)





class resample():
    def __init__(self, wvs_a, wvs_b, fwhm_b):
        self.wvs_a = wvs_a
        self.wvs_b = wvs_b
        self.fwhm_b = fwhm_b

        self.get_transform_matrix()

    
    def get_transform_matrix(self):
        # the transformatrix is applied on an asd spectrum (2000) to transform to avng (425)
        # so it would be y = Ax, A is the transform matrix, x is the 2000 by 1 spectrum
        # So A is 425 by 2000, becasue (425x2000) time (2000x1) = (425x1) which is what we want
        # So each row in A, of length 2000, represents the sensitivity of every avng channel
        # We assume gaussianity, so those are just gaussians.
        
        base_wl = np.array(self.wvs_a)
        target_wl = np.array(self.wvs_b)
        target_fwhm = np.array(self.fwhm_b)

        doTheResample = lambda id: self.spectrumResample(id, base_wl, target_wl, target_fwhm)
        ww = np.array([doTheResample(W) for W in range(len(target_wl))])
        self.transform_matrix = ww
    
    def resample_data(self, y):
        spectrum = np.atleast_2d(y)
        if spectrum.shape[0] == 1:
            spectrum = spectrum.T
        
        resampled_spectrum = np.dot(self.transform_matrix, spectrum)

        return np.squeeze(resampled_spectrum)


    def srf(self, x, mu, sigma):
        """Spectral Response Function """
        u = (x-mu)/abs(sigma)
        y = (1.0/(np.sqrt(2.0*np.pi)*abs(sigma)))*np.exp(-u*u/2.0)
        if y.sum()==0:
            return y
        else:
            return y/y.sum()


    def spectrumResample(self, idx, wl, wl2, fwhm2=10, fill=False):
        """Resample a spectrum to a new wavelength / FWHM.
        I assume Gaussian SRFs"""

        #resampled = np.zeros((wl2.shape[0], 1))
        #for i in range(x.shape[1]):
        resampled = np.array(self.srf(wl, wl2[idx], fwhm2[idx]/2.35482))
            #resampled[:, i] = np.array([self.srf(wl, wi, fwhmi/2.35482)
             #   for wi, fwhmi in zip(wl2, fwhm2)]).reshape((len(wl2)))

        return resampled

