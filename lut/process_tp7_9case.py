#!/usr/bin/env python3


import numpy as np
import sys
import pdb
import matplotlib.pyplot as plt
import re
from resample import resample
import pdb

def read_and_process_file_old(file_path):


    with open(file_path, 'r') as file:
        lines = file.readlines()

    case_indices = [i for i, line in enumerate(lines) if "case index" in line]
    end_indices = [i for i, line in enumerate(lines) if line.strip() == '}']
    cases_data = {}

    # Pair up start and end indices of each case
    end_indices = [i for i, line in enumerate(lines) if line.strip() == '}']
    case_ranges = zip(case_indices, end_indices)

    for case_num, (start_index, end_index) in enumerate(case_ranges, start=1):
        col_names_line_1 = lines[start_index+4].strip().split(',')
        col_names_line_2 = lines[start_index+5].strip().split(',')

        column_names = [name1.strip() + ' ' + name2.strip() for name1, name2 in zip(col_names_line_1, col_names_line_2)]

        data_lines = lines[start_index+6:end_index]
        data = np.genfromtxt(data_lines, delimiter=',', names=column_names)

        cases_data[f"case_{case_num}"] = data

    return cases_data

def read_and_process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    case_indices = [i for i, line in enumerate(lines) if "case index" in line]
    end_indices = [i for i, line in enumerate(lines) if line.strip() == '}']
    cases_data = {}

    case_ranges = zip(case_indices, end_indices)
    seen_cases = set()

    for start_index, end_index in case_ranges:
        match = re.search(r"case index (\d+)", lines[start_index])
        if match:
            case_index = int(match.group(1))
            if case_index in seen_cases:
                continue  # Skip already processed cases
            seen_cases.add(case_index)

            col_names_line_1 = lines[start_index+4].strip().split(',')
            col_names_line_2 = lines[start_index+5].strip().split(',')
            column_names = [name1.strip() + ' ' + name2.strip() for name1, name2 in zip(col_names_line_1, col_names_line_2)]

            data_lines = lines[start_index+6:end_index]
            data = np.genfromtxt(data_lines, delimiter=',', names=column_names)

            cases_data[f"case_{case_index}"] = data
    #pdb.set_trace()
    return cases_data


def process_cases(cases_data):
    # Define the product names, including 'path_multiple_scat' and 'sing_scat'
    product_names = ['grnd_rflt', 'drct_rflt', 'total_rad', '_nat_log_path_trans', 'path_multiple_scat', 'sing_scat', 'ToA_irrad']

    # Predefined mapping of case numbers to albedo groups with resolution
    albedo_groups = {
        '0': [(1, '5cm'), (4, '1cm'), (7, '0.1cm')],
        '0.1': [(2, '5cm'), (5, '1cm'), (8, '0.1cm')],
        '0.5': [(3, '5cm'), (6, '1cm'), (9, '0.1cm')]
    }

    # Initialize a dictionary to store the data for each product and albedo group
    albedo_data = {albedo: {product: [] for product in product_names} for albedo in albedo_groups}
    
    # Dictionary to store seen frequencies with their resolutions
    seen_frequencies = {albedo: {product: {} for product in product_names} for albedo in albedo_groups}

    resolution_map = {'5cm': 1, '1cm': 2, '0.1cm': 3}  # Higher numbers indicate higher resolution

    for albedo_key, group in albedo_groups.items():
        for case_num, resolution in group:
            case_num = case_num - 1 # conssitent with the production of the case numbers in the csv file!
            if False:#case_num==9:
                pdb.set_trace()
            case_key = f"case_{case_num}"
            case_df = cases_data[case_key]

            # Get the frequency column name dynamically
            freq_column_name = case_df.dtype.names[0]
            freq_column = case_df[freq_column_name]

            for product in product_names:
                if product in case_df.dtype.names:  # Check if product is in the original data
                    interest_column = case_df[product]

                    for freq, value in zip(freq_column, interest_column):
                        current_resolution = resolution_map[resolution]
                        if freq not in seen_frequencies[albedo_key][product] or current_resolution > seen_frequencies[albedo_key][product][freq]:
                            seen_frequencies[albedo_key][product][freq] = current_resolution
                            albedo_data[albedo_key][product].append((freq, value))

    # Convert lists to numpy arrays and sort them by frequency
    for albedo, products in albedo_data.items():
        for product, data in products.items():
            # Apply the rdn_in_nm transformation to radiance data
            if product in ['grnd_rflt', 'drct_rflt', 'total_rad', 'path_multiple_scat', 'sing_scat', 'ToA_irrad']:
                transformed_data = [(freq, rdn_in_nm(val, freq)) for freq, val in data]
            elif product == '_nat_log_path_trans':
                transformed_data = [(freq, np.exp(-val)) for freq, val in data]
            else:
                transformed_data = [(freq, val) for freq, val in data]

            # Convert to numpy array and sort by frequency
            structured_array = np.array(transformed_data, dtype=[('Frequency', 'f8'), (product, 'f8')])
            sorted_array = np.sort(structured_array, order='Frequency')
            albedo_data[albedo][product] = sorted_array

    return albedo_data




def rdn_in_nm(rdn, wvn):
    # Conversion from irradiance in W cm^-2 / cm^-1 to W cm^-2 / nm
    # 1. Wavelength (λ) and wavenumber (ν) are related by λ = 1/ν.
    #    To get λ in nanometers, use λ_nm = 10^7 / ν.
    # 2. Differential relation: dλ = -1/ν^2 dν. In nanometers: dλ_nm = -10^7 / ν^2 dν.
    # 3. Conversion Formula: I_λ = I_ν * |dν/dλ_nm|.
    #    Since dλ_nm/dν = -10^7 / ν^2, the conversion factor is |dν/dλ_nm| = ν^2 / 10^7.
    # 4. Apply conversion for each wavenumber (ν) to obtain irradiance in W cm^-2 / nm.
    #    I_λ_nm = I_ν_cm^-1 * (ν^2 / 10^7)
    # Note: This formula considers the squared relationship between ν and λ and unit conversion from cm to nm.

    #wvl = 10**7/wvn
    trans = wvn**2 / 10**7 #10**7/(wvl*10**(-9))**2

    return trans*rdn

def calculate_products(processed_data, infile):

    # let's grab the wavelengths
    freq = processed_data['0']['grnd_rflt']['Frequency']
    wvl = 10**7 / freq
    #pdb.set_trace()
    test_rfls = [0, 0.1, 0.5]
    rfl_1      = test_rfls[1]
    rfl_2      = test_rfls[2]
    widths = 1 # we don't actually need this


    # get SZA and coszen from filename
    match = re.search(r'SZA-([\d]+\.\d+)', infile)
    if match:
        SZA = float(match.group(1))
        coszen = np.cos(np.deg2rad(SZA))
    #pdb.set_trace()


    # get ToA Irrad, is the same in all albedos
    ToA_irrad = processed_data['0']['ToA_irrad']['ToA_irrad'] * 1e6 * coszen / widths / np.pi
    # get t_up_dir, is the same in all albedos
    t_up_dirs = processed_data['0']['_nat_log_path_trans']['_nat_log_path_trans']
    # get direct_ground_reflected amd total_ground_reflected for the 0.1 albedo
    direct_ground_reflected_1 = processed_data['0.1']['drct_rflt']['drct_rflt'] * 1e6
    total_ground_reflected_1 = processed_data['0.1']['grnd_rflt']['grnd_rflt'] * 1e6
    # get total ground reflected from the 0.5 albedo
    total_ground_reflected_2 = processed_data['0.5']['grnd_rflt']['grnd_rflt'] * 1e6
    # get path radiances. We'll need to add the single and multiple together
    path_radiance_1 = (processed_data['0.1']['sing_scat']['sing_scat'] + \
        processed_data['0.1']['path_multiple_scat']['path_multiple_scat']) * 1e6

    path_radiance_2 = (processed_data['0.5']['sing_scat']['sing_scat'] + \
        processed_data['0.5']['path_multiple_scat']['path_multiple_scat']) * 1e6

    path_radiance_0 = processed_data['0']['total_rad']['total_rad'] * 1e6 # This is with the coszen so we will divide later 
        

    # now the 3 albedo method (2 albedo actually):
    direct_flux_1 = direct_ground_reflected_1 * np.pi / rfl_1 / t_up_dirs
    global_flux_1 = total_ground_reflected_1 * np.pi / rfl_1 / t_up_dirs

    global_flux_2 = total_ground_reflected_2 * np.pi / rfl_2 / t_up_dirs

    # Instead of using this I can use the total_rad from the zero reflectance run
    #path_radiance_no_surface = (rfl_2 * path_radiance_1 * global_flux_2 - rfl_1 * path_radiance_2 * global_flux_1) / \
    #                (rfl_2 * global_flux_2 - rfl_1 * global_flux_1)
    
    path_radiance_no_surface_numerator = (rfl_2 * path_radiance_1 * global_flux_2 - rfl_1 * path_radiance_2 * global_flux_1)
    path_radiance_no_surface_denominator = (rfl_2 * global_flux_2 - rfl_1 * global_flux_1)
    path_radiance_no_surface = path_radiance_no_surface_numerator / path_radiance_no_surface_denominator
    


    #path_radiance_no_surface = np.where(np.isnan(path_radiance_no_surface) | (path_radiance_no_surface < 0)\
    #     | (path_radiance_no_surface > 1), 0, path_radiance_no_surface)
    # Diffuse upwelling transmittance
    # I think perhaps we could use either the 0.1 or the 0.5 runs?
    #t_up_difs =  np.pi * (path_radiance_1 - path_radiance_no_surface) / (rfl_1 * global_flux_1)
    t_up_difs_numerator = np.pi * (path_radiance_1 - path_radiance_no_surface)
    t_up_difs_denominator = (rfl_1 * global_flux_1)
    t_up_difs = t_up_difs_numerator / t_up_difs_denominator

    #t_up_difs = np.divide(np.pi * (path_radiance_1 - path_radiance_no_surface), (rfl_1 * global_flux_1), where=(global_flux_1!=0))



    # Spherical Albedo
    # Thanks for Lex Berk's advise we can save the two elements of this quotient 
    #sphalbs = (global_flux_1 - global_flux_2) / \
    #            (rfl_1 * global_flux_1 - rfl_2 * global_flux_2)
    #pdb.set_trace()
    sphalbs_numerator = global_flux_1 - global_flux_2
    sphalbs_denominator = rfl_1 * global_flux_1 - rfl_2 * global_flux_2
    sphalbs = sphalbs_numerator / sphalbs_denominator
    # Set values outside the range [0, 1] or NaN to zero
    # we get singularities so need to dirty fix it. No apparent good solution

    sphalbs = np.where(np.isnan(sphalbs) | (sphalbs < 0) | (sphalbs > 1), np.nan, sphalbs)

    #pdb.set_trace()
    #plt.plot(wvl, sphalbs_numerator, wvl, sphalbs_denominator)
    #plt.legend((['Numerator', 'denominator']))
    #plt.savefig('img2.jpg')

    global_flux_no_surface = global_flux_1*(1.-rfl_1 * sphalbs)
    diffuse_flux_no_surface = global_flux_no_surface - direct_flux_1 * coszen

    t_down_dirs = (direct_flux_1 * coszen / widths / np.pi) / ToA_irrad
    t_down_difs = (diffuse_flux_no_surface / widths / np.pi) / ToA_irrad

    # total transmittance
    transms = (t_down_dirs + t_down_difs) * (t_up_dirs + t_up_difs)
    transup = t_up_dirs + t_up_difs

    # now let's calculate the radiance components
    #L_path = path_radiance_0 / coszen # we have to remember to multiply again in the forward model. I'm open to saving this a different way

    # we are also sorting, i.e., flipping here
    
    wvl = np.flipud(wvl)
    _, unique_indices = np.unique(wvl, return_index=True)
    wvl = wvl[unique_indices]
    

    sphalbs = np.flipud(sphalbs)[unique_indices]
    sphalbs_numerator = np.flipud(sphalbs_numerator)[unique_indices]
    sphalbs_denominator = np.flipud(sphalbs_denominator)[unique_indices]


    transup = np.flipud(transup)[unique_indices]

    t_down_dirs = np.flipud(t_down_dirs)[unique_indices]
    t_down_difs = np.flipud(t_down_difs)[unique_indices]

    path_radiance_0 = np.flipud(path_radiance_0)[unique_indices]

    L_path = path_radiance_no_surface
    L_path = np.flipud(L_path)[unique_indices]
    ToA_irrad = np.flipud(ToA_irrad)[unique_indices]

    #rho_atm = L_path / ToA_irrad # just to make sure it's ok
    
    dir_flux = ToA_irrad * t_down_dirs * transup


    dif_flux = ToA_irrad * t_down_difs * transup

    out_params = {'path_radiance': L_path, 'direct_flux': dir_flux, 'diffuse_flux': dif_flux, 'sphalb_num': sphalbs_numerator, 'sphalb_denom': sphalbs_denominator}
    #pdb.set_trace()
    wvl = {'wavelengths': wvl}

    

    return wvl, out_params


def main(file_path):
    cases_data = read_and_process_file(file_path)
    albedo_processed_data = process_cases(cases_data)
    products = calculate_products(albedo_processed_data, file_path)
    return products


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    products = main(file_path)


'''

if __name__ == "__main__":


    # input is just the tp7 file
    
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    cases_data = read_and_process_file(file_path)
    albedo_processed_data = process_cases(cases_data)

    products = calculate_products(albedo_processed_data, file_path)
    
    
    
    


    # Output or further processing of albedo_processed_data
    print(albedo_processed_data)  # or save to a file, etc.


    emitwvlf = '/scratch/carmon/modtran_luts/projects/multiband/emit_285_wvs_space.txt'
    emit_wl = np.loadtxt(emitwvlf, usecols=1)
    emit_fwhm = np.loadtxt(emitwvlf, usecols=2)

    rs = resample(wvl['wavelengths'], emit_wl, emit_fwhm)
    
    sphalbs_post = rs(sphalbs)
    sphalbs_pre = rs(sphalbs_numerator) / rs(sphalbs_denominator)

    rfl = np.ones_like(L_path) * 0.3
    pdb.set_trace()
    rdn_sphalbs_post = rs(L_path) + (rs(dir_flux)+rs(dif_flux))*rs(rfl)/(1-sphalbs_post*rs(rfl))
    rdn_sphalbs_pre = rs(L_path) + (rs(dir_flux)+rs(dif_flux))*rs(rfl)/(1-sphalbs_pre*rs(rfl))





    relative_diff = (rdn_sphalbs_pre - rdn_sphalbs_post)/rdn_sphalbs_pre

    pdb.set_trace()
    fig, ax1 = plt.subplots()
    fig.tight_layout()
    ax1.plot(emit_wl, rdn_sphalbs_post, 'r-')  # 'g-' is a green solid line
    ax1.plot(emit_wl, rdn_sphalbs_pre, 'g-')
    ax1.set_xlabel('Wavelengths (nm)')
    ax1.set_ylabel('Radiance', color='r')
    ax1.tick_params(axis='y', labelcolor='r')

    # Create a secondary y-axis sharing the same x-axis
    ax2 = ax1.twinx()  
    ax2.plot(emit_wl, relative_diff, 'b-')  # 'b-' is a blue solid line
    ax2.set_ylabel('relative difference', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    # Enable major grid
    ax1.grid(which='major', linestyle='-', linewidth='0.5', color='red')  # Customize major grid
    ax2.grid(which='major', linestyle='-', linewidth='0.25', color='blue')  # Customize major grid
    # Enable minor grid
    ax1.minorticks_on()  # Enable minor ticks, which are required for minor grid
    ax1.grid(which='minor', linestyle=':', linewidth='0.25', color='orange')  # Customize minor grid

    plt.savefig('sphalb_cmpr.jpg', bbox_inches='tight')
    #pdb.set_trace()
    emitwvlf = '/scratch/carmon/modtran_luts/projects/multiband/emit_285_wvs_space.txt'
    emit_wl = np.loadtxt(emitwvlf, usecols=1)
    emit_fwhm = np.loadtxt(emitwvlf, usecols=2)


    # for some reason there are a few dozen multiplies of wavelengths. I'm not sure why.
    # for now let's dirty fix it but in the future we should find out why and fix at the core
    #pdb.set_trace()

    rs = resample(wvl, emit_wl, emit_fwhm)

    # calculate a convolution of the flux:
    #dir_flux_convolved = rs(np.flipud(dir_flux))
    # first convolve stuff then take the product:
    #dir_flux_pre = rs(np.flipud(ToA_irrad)) * rs(np.flipud(t_down_dirs)) * rs(np.flipud(transup))
    #plt.plot(emit_wl, dir_flux_convolved, emit_wl, dir_flux_pre)
    #plt.ylim((0, 20))
    #plt.legend((['post convolved', 'pre convolved']))
    #plt.savefig('img.jpg')
    #pdb.set_trace()
    # now we can just call this and it will convolve.

    # Let's build a TOA radiance for a 30% reflectance:
    rfl = np.ones_like(L_path) * 0.3
    rdn = L_path + (dir_flux+dif_flux)*rfl/(1-sphalbs*rfl)
    rdn_convolved_fluxes = rs(rdn)
    # now convolve and plot
    plt.figure()
    #plt.plot(emit_wl, rdn_convolved_fluxes)

    # now convolve transmmitances and calculate
    rho_atm_convolved = rs(rho_atm)
    t_down_dirs_convolved = rs(t_down_dirs)
    t_down_difs_convolved = rs(t_down_difs)
    transup_convolved = rs(transup)
    ToA_irrad_convolved = rs(ToA_irrad)
    rfl_emit = np.ones_like(rho_atm_convolved)*0.3

    sphalbs_convolved = rs(sphalbs)

    rdn_convolved_transmittances = rho_atm_convolved * ToA_irrad_convolved + ToA_irrad_convolved * (t_down_dirs_convolved + t_down_difs_convolved) * rfl_emit * transup_convolved / (1-sphalbs_convolved * rfl_emit)
    #plt.plot(emit_wl, rdn_convolved_transmittances)

    #pdb.set_trace()
    
    relative_diff = (rdn_convolved_fluxes - rdn_convolved_transmittances)/rdn_convolved_fluxes
    #plt.plot(emit_wl, relative_diff)

    # Create the initial plot with the primary y-axis
    fig, ax1 = plt.subplots()
    fig.tight_layout()
    ax1.plot(emit_wl, rdn_convolved_fluxes, 'r-')  # 'g-' is a green solid line
    ax1.plot(emit_wl, rdn_convolved_transmittances, 'g-')
    ax1.set_xlabel('Wavelengths (nm)')
    ax1.set_ylabel('Radiance (uW / SR / nm / cm^2)', color='r')
    ax1.tick_params(axis='y', labelcolor='r')

    # Create a secondary y-axis sharing the same x-axis
    ax2 = ax1.twinx()  
    ax2.plot(emit_wl, relative_diff, 'b-')  # 'b-' is a blue solid line
    ax2.set_ylabel('relative difference', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    # Enable major grid
    ax1.grid(which='major', linestyle='-', linewidth='0.5', color='red')  # Customize major grid
    ax2.grid(which='major', linestyle='-', linewidth='0.25', color='blue')  # Customize major grid
    # Enable minor grid
    ax1.minorticks_on()  # Enable minor ticks, which are required for minor grid
    ax1.grid(which='minor', linestyle=':', linewidth='0.25', color='orange')  # Customize minor grid

    plt.savefig('img.jpg', bbox_inches='tight')

    #pdb.set_trace()
    
    plt.figure()
    plt.plot(emit_wl, rho_atm_convolved)
    plt.plot(emit_wl, t_down_dirs_convolved)
    plt.plot(emit_wl, t_down_difs_convolved)
    plt.plot(emit_wl, transup_convolved)
    plt.plot(emit_wl, sphalbs_convolved)

    plt.savefig('img2.jpg')

    #pdb.set_trace()

    plt.figure()
    
    wvn = albedo_processed_data['0']['total_rad']['Frequency']
    wvl = 10**7 / wvn

    pdb.set_trace()


    
    
    total_rad0 = albedo_processed_data['0']['total_rad']['total_rad']
    total_rad0_nm = rdn_in_nm(total_rad0, wvn)

    total_radp1 = albedo_processed_data['0.1']['total_rad']['total_rad']
    total_radp1_nm = rdn_in_nm(total_radp1, wvn)

    total_radp5 = albedo_processed_data['0.5']['total_rad']['total_rad']
    total_radp5_nm = rdn_in_nm(total_radp5, wvn)
    plt.plot(wvl, total_rad0, wvl, total_radp1, wvl, total_radp5)
    plt.savefig('img.jpg')

    '''

    

