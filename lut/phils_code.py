#! /usr/bin/env python
#
#  Copyright 2020 California Institute of Technology
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
# Author: Philip G Brodrick, philip.brodrick@jpl.nasa.gov
​
​
​
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from isofit.radiative_transfer.modtran import ModtranRT
from isofit.radiative_transfer.six_s import SixSRT
from isofit.configs import configs
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics
import ray
import glob
import parse
​
​
​
def d2_subset(data,ranges):
    a = data.copy()
    a = a[ranges[0],:]
    a = a[:,ranges[1]]
    return a
​
​
def main():
​
    # Parse arguments
    parser = argparse.ArgumentParser(description="built luts for emulation.")
    parser.add_argument('-config_file', type=str, default='templates/isofit_template.json')
    parser.add_argument('-lut_dir', type=str, default=None)
    parser.add_argument('-keys', type=str, default=['transm', 'rhoatm', 'sphalb'], nargs='+')
    parser.add_argument('-munge_dir', type=str, default='munged')
​
    args = parser.parse_args()
​
    np.random.seed(13)
​
    for key_ind, key in enumerate(args.keys):
        munge_file = os.path.join(args.munge_dir, key + '.npz')
​
        if os.path.isfile(munge_file) is False:
            config = configs.create_new_config(args.config_file)
​
            # Note - this goes way faster if you comment out the Vector Interpolater build section in each of these
            isofit_modtran = ModtranRT(config.forward_model.radiative_transfer.radiative_transfer_engines[0],
                                       config, build_lut = False)
​
​
            if os.path.isdir(os.path.dirname(munge_file) is False):
                os.mkdir(os.path.dirname(munge_file))
​
            if args.lut_dir is not None:
                fileset = glob.glob(os.path.join(args.lut_dir, "*.json"))
                isofit_modtran.files = [os.path.abspath(os.path.splitext(x)[0]) for x in fileset]
​
            isofit_modtran = update_points(isofit_modtran)
            modtran_results = get_obj_res(isofit_modtran, key)
​
            for fn in isofit_modtran.files:
                try:
                    mod_output = isofit_modtran.load_rt(fn)
                except:
                    continue
                sol_irr = mod_output['sol']
                if np.all(np.isfinite(sol_irr)):
                    break
​
            np.savez(munge_file, modtran_results=modtran_results, sol_irr=sol_irr)
​
    modtran_results = None
    for key_ind, key in enumerate(args.keys):
        munge_file = os.path.join(args.munge_dir, key + '.npz')
​
        npzf = np.load(munge_file)
​
        dim1 = int(np.product(np.array(npzf['modtran_results'].shape)[:-1]))
        dim2 = npzf['modtran_results'].shape[-1]
        if modtran_results is None:
            modtran_results = np.zeros((dim1,dim2*len(args.keys)))
        modtran_results[:,dim2*key_ind:dim2*(key_ind+1)] = npzf['modtran_results']
​
        sol_irr = npzf['sol_irr']
​
​
    config = configs.create_new_config(args.config_file)
    isofit_modtran = ModtranRT(config.forward_model.radiative_transfer.radiative_transfer_engines[0],
                               config, build_lut=False)
​
    if args.lut_dir is not None:
        fileset = glob.glob(os.path.join(args.lut_dir, "*.json"))
        isofit_modtran.files = [os.path.abspath(os.path.splitext(x)[0]) for x in fileset]
    isofit_modtran = update_points(isofit_modtran)
    modtran_names = isofit_modtran.lut_names
​
    points = isofit_modtran.points.copy()
​
    print(modtran_results.shape)
    ind = np.lexsort(tuple([points[:,x] for x in range(points.shape[-1])]))
    points = points[ind,:]
    modtran_results = modtran_results[ind,:]
​
    good_data = np.all(np.isnan(modtran_results) == False,axis=1)
​
    modtran_results = modtran_results[good_data,:]
    points = points[good_data,...]
​
    print(modtran_results.shape)
​
    np.savez(os.path.join(args.munge_dir, 'combined_training_data.npz'), modtran_results=modtran_results,
             points=points, keys=args.keys, point_names=modtran_names, modtran_wavelengths=isofit_modtran.wl,
             sol_irr=sol_irr)
​
​
@ray.remote
def read_data_piece(ind, maxind, point, fn, key, resample, obj):
    if ind % 100 == 0:
        print('{}: {}/{}'.format(key, ind, maxind))
    try:
        if resample is False:
            mod_output = obj.load_rt(fn, resample=False)
        else:
            mod_output = obj.load_rt(fn)
        res = mod_output[key]
    except:
        res = None
    return ind, res
​
​
def update_points(obj):
​
    namestring = "IND_$(i)_AERFRAC_2-$(aod)_GNDALT-$(elevation)_H1ALT-$(altitude)_H2OSTR-$(wv)_senzen-$(to_sensor_zenith)_solzen-$(to_solar_zenith)_solzen-$(to_solar_azimuth)_senzen-$(to_sensor_azimuth)"
    obj.lut_names = ['AERFRAC_2','GNDALT','H1ALT','H2OSTR','to_sensor_zenith','to_solar_zenith','to_solar_azimuth','to_sensor_azimuth']
​
    points = np.zeros((len(obj.files), len(obj.lut_names)))
    for _fi, fi in enumerate(obj.files):
        try:
            parsed = [float(x) for x in parse.parse("IND_{}_AERFRAC_2-{}_GNDALT-{}_H1ALT-{}_H2OSTR-{}_senzen-{}_solzen-{}_solzen-{}_senzen-{}",os.path.basename(fi))]
            parsed.pop(0)
            points[_fi,:] = np.array(parsed)
        except:
            print(f'failed to parse file: {fi}')
    obj.points = points
​
    return obj
​
​
def get_obj_res(obj, key, resample=True):
​
    # We don't want the VectorInterpolator, but rather the raw inputs
    ray.init()
​
    if hasattr(obj,'sixs_ngrid_init'):
        results = np.zeros((obj.points.shape[0],obj.sixs_ngrid_init), dtype=float)
    else:
        results = np.zeros((obj.points.shape[0],obj.n_chan), dtype=float)
    objid = ray.put(obj)
    jobs = []
    for ind, (point, fn) in enumerate(zip(obj.points, obj.files)):
        jobs.append(read_data_piece.remote(ind, results.shape[0], point, fn, key, resample, objid))
    rreturn = [ray.get(jid) for jid in jobs]
    for ind, res in rreturn:
        if res is not None:
            try:
                results[ind,:] = res
            except:
                print(f'failed on file: {fn}')
                results[ind,:] = np.nan
        else:
            print(f'failed on file: {fn}')
            results[ind,:] = np.nan
    ray.shutdown()
    return results
​
​
if __name__ == '__main__':
    main()