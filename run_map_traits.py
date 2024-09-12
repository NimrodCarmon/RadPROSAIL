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

import sys
from map_traits import MapTraits
import json
import pdb

def main():
    #pdb.set_trace()
    cnfg_path = sys.argv[1]
    with open(cnfg_path) as config_file:
        config_data = json.load(config_file)
    mapper = MapTraits(config_data)
    debug_mode = False
    mapper.run(debug_mode)
    
    




if __name__=="__main__": main()

