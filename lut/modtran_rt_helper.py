#! /usr/bin/env python3

import numpy as np
import pdb
import scipy


def get_modtran_template_multi_bandmodel(input_dict):
    '''differences compared to former:
    1. We are running 3 different band models. Are requirement for now is to have at least
    3 spectral points per 1nm segment. So the thresholds are: [817, 1826] nm. I.e.,
    we use the 5cm-1 between 350 and 817, the 1cm-1 between 817 and 1826 and the 0.1cm-1 between 1826 and 2520
    2. So we will have this function make a json template based on the above. 
    3. the other change is that we are writing into a csv.
    
    
    '''

def get_modtran_template_quick_tp6(input_dict):
    #This is just to make a tp6 file, no more
        ### from David's requirements:
    # Note that DV and FWHM don't matter if we have a filter file! This is from Lex Berk's email. 
    # He is saying that the convolution with the filter file is done on the highest resolution product
    # in the simulation, i.e., directly on the band model resolution. So really it doesn't matter
    # what we put here! As long as it large enough compared with the band model so it won't through and error.
    dayofyear = 170 #int(input_dict['DOY']) # we will not change this! 
    
    ### DEFAULTS
    # parameters to the spectral card
    # This actually does not matter because we use a filter file and the convolution is applied
    # from the band model on the filter, not the scan.
    DV = 15
    FWHM = 15
    # atmospheric type
    atmosphere_type = str(input_dict['atmosphere_type'])
    ihaze_type = str(input_dict['aerosol_model'])
    fid = str(input_dict['NAME'])
    instrument_altitude = float(input_dict['alt'])
    filter_fname = str(input_dict['filter_file'])
    #pdb.set_trace()
    to_sensor_zenith = float(input_dict['TSZ'])
    ground_elevation = float(input_dict['GNDALT'])
    
 
    aod550 = -float(input_dict['AOT550'])
    h2o = float(input_dict['H2OSTR'])
    solar_zenith = float(input_dict['SZA'])
    relative_azimuth = float(input_dict['RAA'])
    band_model = '05_2013'

    """ Write a MODTRAN template file for use by isofit look up tables

    Args:
        atmosphere_type: label for the type of atmospheric profile to use in modtran
        fid: flight line id (name)
        altitude_km: altitude of the sensor in km
        dayofyear: the current day of the given year
        latitude: acquisition latitude
        longitude: acquisition longitude
        to_sensor_azimuth: azimuth view angle to the sensor, in degrees (AVIRIS convention)
        to_sensor_zenith: azimuth view angle to the sensor, in degrees (MODTRAN convention: 180 - AVIRIS convention)
        gmtime: greenwich mean time
        elevation_km: elevation of the land surface in km
        output_file: location to write the modtran template file to

    """
    # make modtran configuration

    modtran_input = {
        "MODTRANINPUT":{
                "NAME": fid,
                "DESCRIPTION": "",
                "CASE": 0,
                "RTOPTIONS": {
                    "MODTRN": "RT_CORRK_FAST",
                    "LYMOLC": False,
                    "T_BEST": False,
                    "IEMSCT": "RT_TRANSMITTANCE",
                    "IMULT": "RT_NO_MULTIPLE_SCATTER"
                },
                "ATMOSPHERE": {
                    "MODEL": atmosphere_type,
                    "CO2MX": 410.0,
                    "H2OSTR": h2o,
                    "H2OUNIT": "g",
                    "O3STR": 0.3,
                    "O3UNIT": "a"
                },
                "AEROSOLS": {
                    "IHAZE": ihaze_type,
                    "VIS": aod550
                    },
                "GEOMETRY": {
                    "ITYPE": 2,
                    "H1ALT": instrument_altitude,
                    "H2ALT": ground_elevation,
                    "IPARM": 12,
                    "PARM1": relative_azimuth,
                    "PARM2": solar_zenith,
                    "OBSZEN": to_sensor_zenith
                },
                "SURFACE": {
                    "SURFTYPE": "REFL_CONSTANT",
                    "SURREF": 0,
                    "GNDALT": ground_elevation,
                    "NSURF": 1,
                    "SALBFL": ""
                },
                "SPECTRAL": {
                    "V1": 800.0,
                    "V2": 817.0,
                    "DV": DV,
                    "FWHM": FWHM,
                    "YFLAG": "R",
                    "XFLAG": "N",
                    "FLAGS": "NTAA   ",
                    "MLFLX": -1,
                    "LBMNAM": "T",
                    "BMNAME": band_model
                },
                "FILEOPTIONS": {
                    "NOFILE": "FC_TAPE6ONLY",
                    "MSGPRNT": "MSG_NONE",
                    "NOPRNT": 2
                }
            }
    }
    modtran_template = {"MODTRAN": [modtran_input]}
    return modtran_template




def get_modtran_template_one_part(input_dict, h2oopt=False):

    ### from David's requirements:
    # Note that DV and FWHM don't matter if we have a filter file! This is from Lex Berk's email. 
    # He is saying that the convolution with the filter file is done on the highest resolution product
    # in the simulation, i.e., directly on the band model resolution. So really it doesn't matter
    # what we put here! As long as it large enough compared with the band model so it won't through and error.
    dayofyear = 170 #int(input_dict['DOY']) # we will not change this! 
    allow_RT_gt_100 = h2oopt
    ### DEFAULTS
    # parameters to the spectral card
    # This actually does not matter because we use a filter file and the convolution is applied
    # from the band model on the filter, not the scan.
    DV = 15
    FWHM = 15
    # atmospheric type
    atmosphere_type = str(input_dict['atmosphere_type'])
    ihaze_type = str(input_dict['aerosol_model'])
    fid = str(input_dict['NAME'])
    instrument_altitude = float(input_dict['alt'])
    filter_fname = str(input_dict['filter_file'])
    #pdb.set_trace()
    to_sensor_zenith = float(input_dict['TSZ'])
    ground_elevation = float(input_dict['GNDALT'])
    
 
    aod550 = -float(input_dict['AOT550'])
    h2o = float(input_dict['H2OSTR'])
    solar_zenith = float(input_dict['SZA'])
    relative_azimuth = float(input_dict['RAA'])
    band_model = input_dict['band_model']

    """ Write a MODTRAN template file for use by isofit look up tables

    Args:
        atmosphere_type: label for the type of atmospheric profile to use in modtran
        fid: flight line id (name)
        altitude_km: altitude of the sensor in km
        dayofyear: the current day of the given year
        latitude: acquisition latitude
        longitude: acquisition longitude
        to_sensor_azimuth: azimuth view angle to the sensor, in degrees (AVIRIS convention)
        to_sensor_zenith: azimuth view angle to the sensor, in degrees (MODTRAN convention: 180 - AVIRIS convention)
        gmtime: greenwich mean time
        elevation_km: elevation of the land surface in km
        output_file: location to write the modtran template file to

    """
    # make modtran configuration

    modtran_input = {
        "MODTRANINPUT":{
                "NAME": fid,
                "DESCRIPTION": "",
                "CASE": 0,
                "RTOPTIONS": {
                    "MODTRN": "RT_CORRK_FAST",
                    "LYMOLC": False,
                    "T_BEST": False,
                    "IEMSCT": "RT_SOLAR_AND_THERMAL",
                    "IMULT": "RT_DISORT",
                    "DISALB": True,
                    "NSTR": 8,
                    "SOLCON": 0.0
                },
                "ATMOSPHERE": {
                    "MODEL": atmosphere_type,
                    "CO2MX": 420.0,
                    "H2OSTR": h2o,
                    "H2OUNIT": "g",
                    "O3STR": 0.3,
                    "O3UNIT": "a"
                },
                "AEROSOLS": {
                    "IHAZE": ihaze_type,
                    "VIS": aod550
                    },
                "GEOMETRY": {
                    "ITYPE": 2,
                    "H1ALT": instrument_altitude,
                    "H2ALT": ground_elevation,
                    "IPARM": 12,
                    "PARM1": relative_azimuth,
                    "PARM2": solar_zenith,
                    "OBSZEN": to_sensor_zenith
                },
                "SURFACE": {
                    "SURFTYPE": "REFL_CONSTANT",
                    "SURREF": 0,
                    "GNDALT": ground_elevation,
                    "NSURF": 1,
                    "SALBFL": ""
                },
                "SPECTRAL": {
                    "V1": 340.0,
                    "V2": 2520.0,
                    "DV": DV,
                    "FWHM": FWHM,
                    "YFLAG": "R",
                    "XFLAG": "N",
                    "FLAGS": "NT A   ",
                    "BMNAME": band_model,
                    "FILTNM": filter_fname
                },
                "FILEOPTIONS": {
                    "NOFILE": "FC_TAPE6ONLY",
                    "NOPRNT": 2,
                    "CKPRNT": True
                }
            }
    }


    if allow_RT_gt_100 is True:
        #pdb.set_trace()
        modtran_input['MODTRANINPUT']['ATMOSPHERE']['H2OOPT'] = "+"
    modtran_template = {"MODTRAN": [modtran_input]}
    
    return modtran_template


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
                vs = scipy.stats.norm.pdf(ws, wl, sigma)
                vs = vs/vs[int(steps/2)]
                wns = 10000.0/(ws/1000.0)

                fout.write('CENTER:  %6.2f NM   FWHM:  %4.2f NM\n' %
                           (wl, fwhm))

                for w, v, wn in zip(ws, vs, wns):
                    fout.write(' %9.4f %9.7f %9.2f\n' % (w, v, wn))

        return outfile

