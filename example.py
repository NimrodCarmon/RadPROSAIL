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



# Import the model class
# This is just to show how the module works




from prosail2 import Prosail
import pdb
import matplotlib.pyplot as plt
import numpy as np
from BRDF import est_spec
import cProfile
import matplotlib.colors as mcolors

def main():
    #pdb.set_trace()
    # nothing happens, but it gives access to stuff
    p = Prosail()
    wvs = p.wl
#    #def run(self, N, Cab, Car, Cbrown, Cw, Cm, LAI, psoil, hspot, tts, tto, psi, LIDF, outname=None, Py6S=False):
    
    # Common leaf distributions
    Planophile = (1, 0)
    Erectophile = (-1, 0)
    Plagiophile = (0, -1)
    Extremophile = (0, 1)
    Spherical = (-0.35, -0.15)
    
    # psi is the relative azimuth angle
    cnf_template = {'N': 1.5, 'Cab': 40, 'Car': 8, 'Cbrown': 0, 'Cw': 0.01, \
        'Cm': 0.009, 'LAI': 3, 'psoil': 0, 'hspot': 0.5, 'tts': 45, 'tto':1, 'psi':30, 'LIDF': Planophile}
    
    cnf_template = {'N': 2.0, 'Cab': 50, 'Car': 10, 'Cbrown': 0.1, 'Cw': 0.02, \
        'Cm': 0.015, 'LAI': 5, 'psoil': 0.3, 'hspot': 0.8, 'tts': 30, 'tto': 2, 'psi': 45, 'LIDF': 'Erectophile'}
    
    cnf_template = {'N': 1.5, 'Cab': 45, 'Car': 15, 'Cbrown': 0.2, 'Cw': 0.03, \
        'Cm': 0.02, 'LAI': 2, 'psoil': 0.2, 'hspot': 0.5, 'tts': 40, 'tto': 1, 'psi': 60, 'LIDF': 'Spherical'}
    
    cnf_template_corn = {'N': 1.5, 'Cab': 45, 'Car': 8, 'Cbrown': 0.2, 'Cw': 0.03, \
        'Cm': 0.02, 'LAI': 2, 'psoil': 0.2, 'hspot': 0.1, 'tts': 45, 'tto': 30, \
            'psi': 60, 'LIDFa': Erectophile[0], 'LIDFb': Erectophile[1]}


    #spc = p.run(cnf_template_corn)
    #spc = p.run(cnf_template)
    #plt.plot(p.wl, spc, label='Corn')
    #plt.savefig('img.jpg')
    

    # produce a bunch of different Cab levels:
    # Define colormap


    ##### CHLOROPHIL
    colors = ['#D2B48C', '#90EE90', '#008000', '#006400']
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', colors)

    # Create a list of Cab values
    Cab_values = np.arange(20, 100, 7.5)

    for i, Cab in enumerate(Cab_values):
        color=cmap(i / (len(np.arange(20, 100, 10)) - 1))
        cnf_template_corn['Cab'] = Cab
        spc = p.run(cnf_template_corn)
        plt.plot(p.wl, spc, label="Cab = {}".format(Cab), color=color)

    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(Cab_values)
    cbar = plt.colorbar(sm)
    cbar.set_label('Cab')

    plt.ylim((0, 0.3))
    plt.grid()
    plt.xlabel('Wavelengths (nm)')
    plt.ylabel('HDRF')
    #plt.title('Spectrum Plot')
    plt.savefig('img.jpg')
    #pdb.set_trace()

    plt.figure()
    # Define the custom colormap
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', ['#964B00', '#00BFFF'])


    ##### Liquid Water
    # Create a list of Cw values
    Cw_values = np.arange(0.01, 0.1, 0.01)

    # Iterate over Cw values
    for i, Cw in enumerate(Cw_values):
        cnf_template_corn['Cw'] = Cw
        spc = p.run(cnf_template_corn)
        plt.plot(p.wl, spc, color=cmap(i / (len(Cw_values) - 1)))

    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(Cw_values)
    cbar = plt.colorbar(sm)
    cbar.set_label('Cw')
    plt.ylim((0, 0.3))
    plt.grid()
    plt.xlabel('Wavelengths (nm)')
    plt.ylabel('HDRF')
    #plt.title('Spectrum Plot')
    plt.savefig('Cw.jpg')
    #pdb.set_trace()


    ###### BROWN  PIGMENT
    plt.figure()
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', ['#008000', '#A52A2A'])
    # Create a list of Cw values
    Cbrown_values = np.arange(0.01, 1, 0.05)

    # Iterate over Cw values
    for i, Cbrown in enumerate(Cbrown_values):
        cnf_template_corn['Cbrown'] = Cbrown
        spc = p.run(cnf_template_corn)
        plt.plot(p.wl, spc, color=cmap(i / (len(Cbrown_values) - 1)))

    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(Cbrown_values)
    cbar = plt.colorbar(sm)
    cbar.set_label('Cbrown')
    plt.ylim((0, 0.3))
    plt.grid()
    plt.xlabel('Wavelengths (nm)')
    plt.ylabel('HDRF')
    #plt.title('Spectrum Plot')
    plt.savefig('Cbrown.jpg')
    #pdb.set_trace()

    #### LAI
    plt.figure()
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', ['#00FF00', '#006400'])

    # Create a list of Cw values
    LAI_values = np.arange(1, 10, 0.5)

    # Iterate over Cw values
    for i, LAI in enumerate(LAI_values):
        cnf_template_corn['LAI'] = LAI
        spc = p.run(cnf_template_corn)
        plt.plot(p.wl, spc, color=cmap(i / (len(LAI_values) - 1)))

    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(LAI_values)
    cbar = plt.colorbar(sm)
    cbar.set_label('LAI')
    plt.ylim((0,0.3))
    plt.grid()
    plt.xlabel('Wavelengths (nm)')
    plt.ylabel('HDRF')
    #plt.title('Spectrum Plot')
    plt.savefig('LAI.jpg')
    print('done')
    pdb.set_trace()




    results = []
    for lidf, typename in zip([Planophile, Erectophile, Plagiophile, Extremophile, Spherical], ['Planophile', 'Erectophile', 'Plagiophile', 'Extremophile', 'Spherical']):
        cnf_template['LIDF'] = lidf
        spc = p.run(cnf_template)
        plt.plot(p.wl, spc, label=typename)
        results.append(spc)
    

    # get a single output:
    #profiler = cProfile.Profile()
    #profiler.enable()
    results1 = p.run(cnf_template)
    #profiler.disable()
    #profiler.print_stats()
    #p.run(1.5, 40, 8, 0, 0.01, 0.009, 1, 3, 0.01, 30, 10, 10, p.Planophile)
    #profile_result = cProfile.run('runit()', globals(), locals())
    #print("Method return value:", profile_result)
    pdb.set_trace()
    
    cnf = cnf_template.copy()
    
    var_name = 'N'
    var_range = list(np.arange(0, 5, 0.1))
    #tto_range = list(np.arange(0, 20))
    results = []
    for val in var_range:
        cnf[var_name] = val
        print(cnf[var_name])
        results.append(p.run(cnf))


    wvs = p.wl
    results = np.array(results).T
    #cnfs = [cnf.update(tts=value) for value in np.arange(30, 60)]
    #plt.plot(wvs, results); plt.show()
    pdb.set_trace()
    try_rho = results[:, 0]
    brdf_adj = est_spec(try_rho, 2, 30, 5, 30, 50, 10)
    plt.plot(wvs, try_rho, wvs, brdf_adj); plt.show()
    #results = p.run(cnf)
    print(results)

    # Results ready for use with Py6S
    #results2 = p.run(1.5, 40, 8, 0, 0.01, 0.009, 1, 3, 0.01, 30, 10, 10, p.Planophile, Py6S=True)
    #print(results2)
    pdb.set_trace()

    # Use these results with Py6S by running something like:
    # s = SixS()
    # s.ground_reflectance = GroundReflectance.HomogeneousLambertian(results2)
    # s.run()


def runit(p):
    Planophile = (1, 0)
    cnf_template = {'N': 1.5, 'Cab': 40, 'Car': 8, 'Cbrown': 0, 'Cw': 0.01, \
        'Cm': 0.009, 'LAI': 1, 'psoil': 3, 'hspot': 0.01, 'tts': 30, 'tto':10, 'psi':10, 'LIDF': Planophile}
    return p.run(cnf_template)


def plotspc(results):
    plt.plot(results[:,0], results[:,1])
    return 1


    
    return wvs

if __name__=='__main__': main()