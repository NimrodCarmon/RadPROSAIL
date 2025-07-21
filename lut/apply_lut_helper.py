from spectral.io import envi
import scipy as sp
from scipy import stats
import numpy as np
import glob
import pdb

# a couple of helper functions

def define_spectral_info(basedir, wvs_file, wvs_cnfg):
    #pdb.set_trace()
    readfromfile = wvs_cnfg['from_file']
    if readfromfile is True:

        file2read = wvs_cnfg['wvs_file']
        data = np.loadtxt(file2read, usecols=(1, 2))
        wl = data[:,0]
        fwhm = data[:,1]

    elif False:#:readfromfile is True
        # option to read from a radiance or reflectance file
        hdrfile = '/home/nimrod/proj/emit_correction/remote/EMIT_L1B_RAD_001_20220818T114428_2223008_003_radiance.hdr'
        #pdb.set_trace()
        #rdn = glob.glob(remotedir+'/*rdn')[0]
        #rdnfi = envi.open(rdn+'.hdr')
        rdnfi = envi.open(hdrfile)
        hdr = rdnfi.metadata.copy()
        wl = hdr['wavelength']
        wl = np.array([float(w) for w in wl])
        fwhm = hdr['fwhm']
        fwhm = np.array([float(w) for w in fwhm])

    else: # we'll just do the 0.5nm grid here with 1.5nm fwhm
        file2read = wvs_file
        minwv = wvs_cnfg['min']
        maxwv = wvs_cnfg['max']+1 # always do the plus one if you want the full range min max
        dv = wvs_cnfg['DV']
        fwhm = wvs_cnfg['FWHM']
        
        wl = np.arange(minwv,maxwv,dv)
        fwhm = np.ones(wl.shape)*fwhm

    wlf = file2read
    sp.savetxt(wlf, sp.c_[sp.ones(len(wl)), wl, fwhm], fmt='%12.8f') # sp.c_ slices arrays across rows

    return wl, fwhm, wlf


def wl2flt(wavelengths: np.array, fwhms: np.array, outfile: str) -> str:
        """Helper function to generate Gaussian distributions around the
        center wavelengths.

        Args:
            wavelengths: wavelength centers
            fwhms: full width at half max
            outfile: file to write to

        """
        #import pdb; pdb.set_trace()
        outfile = outfile.split('.')[0]+'_filter.txt'
        sigmas = fwhms/2.355
        span = 2.0 * np.abs(wavelengths[1]-wavelengths[0])  # nm
        steps = 101

        with open(outfile, 'w') as fout:

            fout.write('Nanometer data for sensor\n')
            for wl, fwhm, sigma in zip(wavelengths, fwhms, sigmas):

                ws = wl + np.linspace(-span, span, steps)
                vs = stats.norm.pdf(ws, wl, sigma)
                vs = vs/vs[int(steps/2)]
                wns = 10000.0/(ws/1000.0)

                fout.write('CENTER:  %6.2f NM   FWHM:  %4.2f NM\n' %
                           (wl, fwhm))

                for w, v, wn in zip(ws, vs, wns):
                    fout.write(' %9.4f %9.7f %9.2f\n' % (w, v, wn))

        return outfile
