

import pdb
import numpy as np
import re
import sys
import os
import json
import re
import warnings
# if debugging turn this off
warnings.filterwarnings("ignore")

def main():
    # this is just for debugging
    infile = sys.argv[1]

    out = load_chn(infile)


def how_many_bands(file_path):

    num_bands = 0
    previous_band = 0
    number_pattern = re.compile(r'^[1-9]\d{2,}(\.\d+)?.*$')

    with open(file_path, 'r') as file:
        for line in file:

            line = line.strip()  # Remove leading/trailing whitespaces
            if not line or not number_pattern.match(line):
                continue  # Skip empty lines or lines not matching the pattern
            
            current_band = float(line.split()[0])  # Assumes numbers are in the first column
            
            if current_band < previous_band:
                break
            if current_band > 300 and current_band != previous_band:
                num_bands += 1
                previous_band = current_band

    return num_bands


# need to complete the dict in so it replaces the self object here
def load_chn_single(infile, multipart):
    """Load a '.chn' output file and parse critical coefficient vectors.

        These are:
            * wl      - wavelength vector
            * sol_irr - solar irradiance
            * sphalb  - spherical sky albedo at surface
            * transm  - diffuse and direct irradiance along the
                        sun-ground-sensor path
            * transup - transmission along the ground-sensor path only

        If the "multipart transmittance" option is active, we will use
        a combination of three MODTRAN runs to estimate the following
        additional quantities:
            * t_down_dir - direct downwelling transmittance
            * t_down_dif - diffuse downwelling transmittance
            * t_up_dir   - direct upwelling transmittance
            * t_up_dif   - diffuse upwelling transmittance

        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        Be careful with these! They are to be used only by the
        modtran_tir functions because MODTRAN must be run with a
        reflectivity of 1 for them to be used in the RTM defined
        in radiative_transfer.py.

        * thermal_upwelling - atmospheric path radiance
        * thermal_downwelling - sky-integrated thermal path radiance
            reflected off the ground and back into the sensor.

        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        We parse them one wavelength at a time."""
    multipart_transmittance = multipart
    test_rfls = [0, 0.1, 0.5]
    match = re.search(r'SZA-([\d]+\.\d+)', infile)

    if match:
        SZA = float(match.group(1))
        coszen = np.cos(np.deg2rad(SZA))

    nwl = how_many_bands(infile)

    with open(infile) as f:
        sols, transms, sphalbs, wls, rhoatms, transups = \
            [], [], [], [], [], []
        t_down_dirs, t_down_difs, t_up_dirs, t_up_difs = [],[],[],[]
        grnd_rflts_1, drct_rflts_1, grnd_rflts_2, drct_rflts_2 = \
            [], [], [], []
        transm_dirs, transm_difs, widths = [],[],[]
        lp_0, lp_1, lp_2 = [],[],[]
        thermal_upwellings, thermal_downwellings = [], []
        lines = f.readlines()
        nheader = 5
        #pdb.set_trace()
        # Mark header and data segments
        case = -np.ones(nheader*3+nwl*3)
        case[nheader:(nheader+nwl)] = 0
        case[(nheader*2+nwl):(nheader*2+nwl*2)] = 1
        case[(nheader*3+nwl*2):(nheader*3+nwl*3)] = 2

        for i, line in enumerate(lines):

            # exclude headers
            if case[i] < 0:
                continue
            
            try:
                # Columns 1 and 2 can touch for large datasets.
                # Since we don't care about the values, we overwrite the
                # character to the left of column 1 with a space so that
                # we can use simple space-separated parsing later and
                # preserve data indices.
                line = line[:17]+' '+line[18:]

                # parse data out of each line in the MODTRAN output
                toks = re.findall(r"[\S]+", line.strip())
                wl, wid = float(toks[0]), float(toks[8])  # nm
                solar_irr = float(toks[18]) * 1e6 * \
                    np.pi / wid / coszen  # uW/nm/sr/cm2
                rdnatm  = float(toks[4]) * 1e6  # uW/nm/sr/cm2
                rhoatm  = rdnatm * np.pi / (solar_irr * coszen)
                sphalb  = float(toks[23])
                A_coeff = float(toks[21])
                B_coeff = float(toks[22])
                transm  = A_coeff + B_coeff
                transup = float(toks[24])

                # Be careful with these! See note in function comments above
                thermal_emission = float(toks[11])
                thermal_scatter = float(toks[12])
                thermal_upwelling = (thermal_emission + thermal_scatter) / \
                    wid * 1e6  # uW/nm/sr/cm2

                # Be careful with these! See note in function comments above
                # grnd_rflt already includes ground-to-sensor transmission
                grnd_rflt = float(toks[16]) * 1e6  # ground reflected radiance (direct+diffuse+multiple scattering)
                drct_rflt = float(toks[17]) * 1e6  # same as 16 but only on the sun->surface->sensor path (only direct)
                path_rdn  = float(toks[14]) * 1e6 + float(toks[15]) * 1e6  # The sum of the (1) single scattering and (2) multiple scattering
                thermal_downwelling = grnd_rflt / wid # uW/nm/sr/cm2
            except:
                pdb.set_trace()

            if case[i] == 0:
                try:
                    sols.append(solar_irr)      # solar irradiance
                    transms.append(transm)      # total transmittance
                    sphalbs.append(sphalb)      # spherical albedo
                    rhoatms.append(rhoatm)      # atmospheric reflectance
                    transups.append(transup)    # upwelling direct transmittance
                    transm_dirs.append(A_coeff) # total direct transmittance
                    transm_difs.append(B_coeff) # total diffuse transmittance
                    widths.append(wid)          # channel width in nm
                    lp_0.append(path_rdn)       # path radiance of zero surface reflectance
                    thermal_upwellings.append(thermal_upwelling)
                    thermal_downwellings.append(thermal_downwelling)
                    wls.append(wl) #wavelengths in nm
                except:
                    pdb.set_trace()

            elif case[i] == 1:
                try:
                    grnd_rflts_1.append(grnd_rflt) #total ground reflected radiance
                    drct_rflts_1.append(drct_rflt) #direct path ground reflected radiance
                    lp_1.append(path_rdn) #path radiance (sum of single and multiple scattering)
                except:
                    pdb.set_trace()

            elif case[i] == 2:
                try:
                    grnd_rflts_2.append(grnd_rflt) #total ground reflected radiance
                    drct_rflts_2.append(drct_rflt) #direct path ground reflected radiance
                    lp_2.append(path_rdn) #path radiance (sum of single and multiple scattering)
                except:
                    pdb.set_trace()

    if multipart_transmittance:
        '''
            This implementation is following Gaunter et al. (2009) (DOI:10.1080/01431160802438555),
            and modified by Nimrod Carmon. It is called the "2-albedo" method, referring to running
            modtran with 2 different surface albedos. The 3-albedo method is similar to this one with
            the single difference where the "path_radiance_no_surface" variable is taken from a
            zero-surface-reflectance modtran run instead of being calculated from 2 modtran outputs.
            There are a few argument as to why this approach is beneficial:
            (1) for each grid point on the lookup table you sample modtran 2 or 3 times, i.e. you get
            2 or 3 "data points" for the atmospheric parameter of interest. This in theory allows us
            to use a lower band model resolution modtran run, which is much faster, while keeping
            high accuracy. Currently we have the 5 cm-1 band model resolution configured.
            The second advantage is the possibility to use the decoupled transmittance products to exapnd
            the forward model and account for more physics e.g. shadows \ sky view \ adjacency \ terrain etc.

        '''
        
        t_up_dirs = np.array(transups)
        direct_ground_reflected_1   = np.array(drct_rflts_1)
        total_ground_reflected_1    = np.array(grnd_rflts_1)
        direct_ground_reflected_2   = np.array(drct_rflts_2)
        total_ground_reflected_2    = np.array(grnd_rflts_2)
        path_radiance_1       = np.array(lp_1)
        path_radiance_2       = np.array(lp_2)
        TOA_Irad   = np.array(sols) * coszen / np.pi
        rfl_1      = test_rfls[1]
        rfl_2      = test_rfls[2]
        mus        = coszen


        direct_flux_1 = direct_ground_reflected_1 * np.pi / rfl_1 / t_up_dirs
        global_flux_1 = total_ground_reflected_1 * np.pi / rfl_1 / t_up_dirs
        diffuse_flux_1 = global_flux_1 - direct_flux_1 # diffuse flux

        global_flux_2 = total_ground_reflected_2 * np.pi / rfl_2 / t_up_dirs

        # Instead of using this I can use the total_rad from the zero reflectance run
        path_radiance_no_surface = (rfl_2 * path_radiance_1 * global_flux_2 - \
                                    rfl_1 * path_radiance_2 * global_flux_1) / \
                        (rfl_2 * global_flux_2 - rfl_1 * global_flux_1)

        # Diffuse upwelling transmittance
        # I think perhaps we could use either the 0.1 or the 0.5 runs?
        t_up_difs =  np.pi * (path_radiance_1 - path_radiance_no_surface) / \
                                (rfl_1 * global_flux_1)

        # Spherical Albedo
        sphalbs = (global_flux_1 - global_flux_2) / \
                    (rfl_1 * global_flux_1 - rfl_2 * global_flux_2)
        direct_flux_radiance = direct_flux_1/mus

        global_flux_no_surface = global_flux_1*(1.-rfl_1 * sphalbs)
        diffuse_flux_no_surface = global_flux_no_surface - direct_flux_radiance * coszen

        t_down_dirs = (direct_flux_radiance * coszen / widths / np.pi) / TOA_Irad
        t_down_difs = (diffuse_flux_no_surface / widths / np.pi) / TOA_Irad

        # total transmittance
        transms = (t_down_dirs + t_down_difs) * (t_up_dirs + t_up_difs)
        transup = t_up_dirs + t_up_difs

    if multipart_transmittance is False:
        # we need consistency in the output for later stages

        t_down_dirs = [1]*len(wls)
        t_down_difs = [1]*len(wls)
        t_up_dirs = [1]*len(wls)
        t_up_difs = [1]*len(wls)


    #params = [np.array(i) for i in [wls, sols, rhoatms, transms, sphalbs, transups,
    #                                t_down_dirs, t_down_difs, t_up_dirs, t_up_difs,
    #                                thermal_upwellings, thermal_downwellings]]

    out_params = {'wls': wls, 'sols': sols, 'rhoatm':rhoatms, 'transm':transms, 'sphalb':sphalbs, \
        't_down_dir':t_down_dirs, 't_down_dif':t_down_difs, 't_up_dir':t_up_dirs, 't_up_dif':t_up_difs,\
            'transup': transup, 'thermal_upwellings':thermal_upwellings, 'thermal_downwellings':thermal_downwellings}
    #return tuple(params)
    #['rhoatm', 'transm', 'sphalb', 'transup', 't_down_dir', 't_down_dif', 't_up_dir', 't_up_dif']
    return out_params



def load_chn(infile):
    # we have the coszen in the filename
    # Let's say that this script figures out the parameter space and the output space together
    # So then it writes out everything, instead of having the calling script do the name parsing, ok?
    multimodtran = False

    json_fname = os.path.splitext(infile)[0] + '.json'
    f1 = open(json_fname)
    ds = json.load(f1)

    try:
        nlos = ds['MODTRAN'][0]['MODTRANINPUT']['GEOMETRY']['NLOS']
    except:
        nlos = 1

    with open(infile) as f:
        # figure out stuff from first case, first LOS
        sols, transms, sphalbs, wls, rhoatms, transups = \
            [], [], [], [], [], []
        t_down_dirs, t_down_difs, t_up_dirs, t_up_difs = [],[],[],[]
        grnd_rflts_1, drct_rflts_1, grnd_rflts_2, drct_rflts_2 = \
            [], [], [], []
        transm_dirs, transm_difs, widths = [],[],[]
        lp_0, lp_1, lp_2 = [],[],[]
        thermal_upwellings, thermal_downwellings = [], []
        lines = f.readlines()
        nheader = 5
        #pdb.set_trace()
        lines2 = [np.array(re.findall(r"[\S]+", line.strip())) for line in lines]
        wvs = [line[0] for line in lines2 if len(line)>0]
        wvs2 = [wavelength for wavelength in wvs if wavelength[0:2].isnumeric()]
        wvss = np.array(list(set(wvs2)), dtype=float)
        wvss = np.sort(wvss)
        nwl = len(wvss)
        # now we will have a chunk of LOSs and need to figure out
        # when it ends and moves to the next case
        #pdb.set_trace()
        gap_idx = [idx for idx,line in enumerate(lines2) if len(line)==0]

        
        case0_los0 = lines2[gap_idx[0]+nheader : gap_idx[0]+nheader+nwl]

        if nlos>1:
            case0_mlos = lines2[gap_idx[0]+nheader : gap_idx[1]]
            case0_los_allbutfirst = lines2[gap_idx[0]+nheader+nwl : gap_idx[1]]
            case1_10per_rfl = lines2[gap_idx[1]+nheader : gap_idx[2]]
            case2_50per_rfl = lines2[gap_idx[2]+nheader:]

        # get parameters for multipart:
        # from case 0 los 0 get:
        # -1. coszen, assuming standard naming convensions for json\chn files
        coszen = np.cos(np.deg2rad(float(infile.strip('.chn').split('/')[-1].split('_')[0].split('-')[1])))
        # 0. get wavelength and width
        wl = np.asarray([line[0] for line in case0_los0], dtype=float)
        wid = np.asarray([line[8] for line in case0_los0], dtype=float)
        widths = wid
        # 1. solar irradiance from tok[18] * 1e6 * np.pi / wid / coszen
        solar_irr = np.asarray([line[18] for line in case0_los0], dtype=float) * 1e6 * np.pi / wid / coszen
        sols = solar_irr
        # 2. path radiance, rdnatm,, tok[4] * 1e6
        rdnatm  = np.asarray([line[4] for line in case0_los0], dtype=float) * 1e6
        # calculate atmospheric reflectance via rdnatm * np.pi / (solar_irr * coszen)
        rhoatm  = rdnatm * np.pi / (solar_irr * coszen)
        rhoatms = rhoatm
        # 3. spharical albedo from tok[23]
        sphalb  = np.asarray([line[23] for line in case0_los0], dtype=float)
        sphalbs = sphalb
        # 4. A_coeffs = toks[21]
        A_coeff = np.asarray([line[21] for line in case0_los0], dtype=float)
        # 5. B_coeffs = toks[22]
        B_coeff = np.asarray([line[22] for line in case0_los0], dtype=float)
        # calculate the sum of 4 and 5, transm = A+B
        transm  = A_coeff + B_coeff 
        # 6.  calculate upwards transmmitance, in tok[24], not this is for the 180 obs angle case
        transup = np.asarray([line[24] for line in case0_los0], dtype=float)
        transups = transup


    if multimodtran: 
        ''' 
            This implementation is following Gaunter et al. (2009) (DOI:10.1080/01431160802438555),
            and modified by Nimrod Carmon. It is called the "2-albedo" method, referring to running 
            modtran with 2 different surface albedos. The 3-albedo method is similar to this one with 
            the single difference where the "path_radiance_no_surface" variable is taken from a
            zero-surface-reflectance modtran run instead of being calculated from 2 modtran outputs.
            There are a few argument as to why this approach is beneficial:
            (1) for each grid point on the lookup table you sample modtran 2 or 3 times, i.e. you get 
            2 or 3 "data points" for the atmospheric parameter of interest. This in theory allows us 
            to use a lower band model resolution modtran run, which is much faster, while keeping 
            high accuracy. Currently we have the 5 cm-1 band model resolution configured.
            The second advantage is the possibility to use the decoupled transmittance products to exapnd 
            the forward model and account for more physics e.g. shadows \ sky view \ adjacency \ terrain etc.
                
        '''

        # now let's grab the relevant quantities from cases 1 and 2
        #pdb.set_trace()
        grnd_rflts_1 = np.asarray([line[16] for line in case1_10per_rfl], dtype=float) * 1e6
        drct_rflts_1 = np.asarray([line[17] for line in case1_10per_rfl], dtype=float) * 1e6
        lp_1 = np.asarray([float(line[14])+float(line[15]) for line in case1_10per_rfl], dtype=float) * 1e6

        grnd_rflts_2 = np.asarray([line[16] for line in case2_50per_rfl], dtype=float) * 1e6
        drct_rflts_2 = np.asarray([line[17] for line in case2_50per_rfl], dtype=float) * 1e6
        lp_2 = np.asarray([float(line[14])+float(line[15]) for line in case2_50per_rfl], dtype=float) * 1e6

        # Finally, quantities from the MLOS run
        # here we'll grab the directional stuff, so the path radiance and upward transmmitance
        transm_up = np.asarray(np.array_split(np.asarray([line[24] for line in case0_los_allbutfirst], dtype=float), nlos-1))
        rdns = np.asarray(np.array_split(np.asarray([line[4] for line in case0_los_allbutfirst], dtype=float), nlos-1)) * 1e6

        #pdb.set_trace()
        t_up_dirs = np.array(transups) 
        direct_ground_reflected_1   = np.array(drct_rflts_1) 
        total_ground_reflected_1    = np.array(grnd_rflts_1) 
        direct_ground_reflected_2   = np.array(drct_rflts_2) 
        total_ground_reflected_2    = np.array(grnd_rflts_2) 
        path_radiance_1       = np.array(lp_1) 
        path_radiance_2       = np.array(lp_2) 
        TOA_Irad   = np.array(sols) * coszen / np.pi
        rflstemplate = np.ones(TOA_Irad.shape)
        rfl_1      = rflstemplate*0.1#self.test_rfls[1]
        rfl_2      = rflstemplate*0.5#self.test_rfls[2]
        mus        = coszen
        
        direct_flux_1 = direct_ground_reflected_1 * np.pi / rfl_1 / t_up_dirs 
        global_flux_1 = total_ground_reflected_1 * np.pi / rfl_1 / t_up_dirs 
        diffuse_flux_1 = global_flux_1 - direct_flux_1 # diffuse flux

        global_flux_2 = total_ground_reflected_2 * np.pi / rfl_2 / t_up_dirs

        path_radiance_no_surface = (rfl_2 * path_radiance_1 * global_flux_2 - rfl_1 * path_radiance_2 * global_flux_1) / (rfl_2 * global_flux_2 - rfl_1 * global_flux_1)

        # Diffuse upwelling transmittance
        t_up_difs =  np.pi * (path_radiance_1 - path_radiance_no_surface) / (rfl_1 * global_flux_1) 

        # Spherical Albedo
        sphalbs = (global_flux_1 - global_flux_2) / (rfl_1 * global_flux_1 - rfl_2 * global_flux_2)
        direct_flux_radiance = direct_flux_1/mus

        global_flux_no_surface = global_flux_1*(1.-rfl_1 * sphalbs) 
        diffuse_flux_no_surface = global_flux_no_surface - direct_flux_radiance * coszen
        
        t_down_dirs = (direct_flux_radiance * coszen / widths / np.pi) / TOA_Irad
        t_down_difs = (diffuse_flux_no_surface / widths / np.pi) / TOA_Irad
        
        # total transmittance
        transms = (t_down_dirs + t_down_difs) * (t_up_dirs + t_up_difs)
    #import pdb; pdb.set_trace()
    l_atm = np.array(rhoatms) * (np.array(sols)*coszen) / np.pi
    bnames = ['wl', 'sols', 'l_atm', 'rhoatm', 'transms', 'sphalbs', 'transups', 't_down_dirs', 't_down_difs', 
        't_up_dirs', 't_up_difs']
    

    # the regular output dict    
    params = {}
    outputlist = [wls, sols, l_atm, rhoatms, transms, sphalbs, transups,
                                    t_down_dirs, t_down_difs, t_up_dirs, t_up_difs]

    for i, name in enumerate(bnames):
        params[name] = outputlist[i]

        # parse name to get parameter space:
    point1 = {}
    gridname = infile.strip('.chn').split('/')[-1].split('_')

    
    for lut_dim in gridname:
        try:
            name=lut_dim.split('-')[0]
            val = lut_dim.split('-')[1]
            point1[str(name)] = float(val)
            if name=='SZA':
                coszen = np.round(np.cos(np.deg2rad(float(val))), 3)
        except:
            pass


    # now we want to have direct upwards transmmitance and path radiance to all the LOSs.
    # But we need their coordinates, and that's in the json file


    mlos = ds['MODTRAN'][0]['MODTRANINPUT']['GEOMETRY']['MLOS'][1:] # we don't take the first one
    
    for point, transmmitance, path in zip(mlos, transm_up, rdns):
        point['t_up_direct'] = transmmitance
        point['path'] = path
    
    param_space = []
    response_space = []
    for los in mlos:
        space = point1.copy()
        space['TSZ'] = los['OBSZEN']
        space['RAA'] = los['AZ_INP']
        param_space.append(space)
        response = params.copy()
        try:
            response['l_atm'] = los['path']
        except:
            pdb.set_trace()
        response['t_up_dirs'] = los['t_up_direct']
        response_space.append(response)

    #pdb.set_trace()
    return param_space, response_space


if __name__=="__main__": main()