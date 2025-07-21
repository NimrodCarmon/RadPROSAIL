#!/usr/bin/env python3

import numpy as np
import pdb
import matplotlib.pyplot as plt

class resample():
    def __init__(self, wvs_a, wvs_b, fwhm_b):
        self.wvs_a = wvs_a
        self.wvs_b = wvs_b
        self.fwhm_b = fwhm_b
        #pdb.set_trace()
        self.get_transform_matrix()

    
    def get_transform_matrix(self):
        # the transformatrix is applied on an asd spectrum (2000) to transform to avng (425)
        # so it would be y = Ax, A is the transform matrix, x is the 2000 by 1 spectrum
        # So A is 425 by 2000, becasue (425x2000) time (2000x1) = (425x1) which is what we want
        # So each row in A, of length 2000, represents the sensitivity of every avng channel
        # We assume gaussianity, so those are just gaussians.
        
        base_wl = np.array(self.wvs_a)
        self.base_wl = base_wl
        target_wl = np.array(self.wvs_b)
        self.target_wl = target_wl
        target_fwhm = np.array(self.fwhm_b)

        doTheResample = lambda id: self.spectrumResample(id, base_wl, target_wl, target_fwhm)
        ww = np.array([doTheResample(W) for W in range(len(target_wl))])
        self.transform_matrix = ww
    
    def __call__(self, y):
        # Convert input to 2D array and transpose if necessary
        spectrum = np.atleast_2d(y)
        if spectrum.shape[0] == 1:
            spectrum = spectrum.T

        # Initialize an output array
        resampled_spectrum = np.zeros(self.transform_matrix.shape[0])

        # Apply SRF only to non-NaN parts of the spectrum
        for i in range(self.transform_matrix.shape[0]):
            # Get the current row of the transform matrix
            transform_row = self.transform_matrix[i, :]

            # Identify valid (non-NaN, non-inf, non--inf) elements in the spectrum
            valid_indices = np.where(np.isfinite(spectrum))[0]

            # Perform convolution using only the valid elements
            resampled_spectrum[i] = np.dot(transform_row[valid_indices], spectrum[valid_indices])

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