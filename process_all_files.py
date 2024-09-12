import os
import json
from spectral.io import envi
import pdb
import numpy as np

remote_dir = "/scratch/carmon/lcluc_fire/remote"
output_folder = "/scratch/carmon/lcluc_fire/remote"
prosail_database_file = "/scratch/carmon/prosail/ProSAIL/dataSpec_P5.csv"

# Collect unique dates
dates = set()

for filename in os.listdir(remote_dir):
    if filename.startswith("ls8") and filename.endswith("_rfl.hdr"):
        date = filename[3:11]  # Extracting yyyymmdd
        dates.add(date)

# Iterate through unique dates and process
for date in sorted(dates):
    traits_hdr_file = os.path.join(remote_dir, f"ls8{date}_traits.hdr")
    if os.path.exists(traits_hdr_file):
        # check if file was processed:
        #pdb.set_trace()
        print('checking '+traits_hdr_file)
        img = envi.open(traits_hdr_file)
        mmap = img.open_memmap()
        random_line = mmap[:,3000,0]
        
        # bad files would show all zeros
        if np.all(random_line==0):
            del mmap
            del img
            # so this is a bad file and needs to run again
            #pdb.set_trace()
            pass
        else:
            del mmap
            del img
            #this file is good
            continue

    # Build config
    config = {
        "input": {
            "input_rfl": os.path.join(remote_dir, f"ls8{date}_rfl.hdr"),
            "input_uncert": "path_to_uncert",
            "input_obs": os.path.join(remote_dir, f"ls8{date}_obs.hdr"),
        },
        "output": {
            "output_folder": output_folder
        },
        "prosail_database_file": prosail_database_file
    }

    # Save config to file
    config_file_path = os.path.join(remote_dir, f"ls8{date}_config.json")
    with open(config_file_path, 'w') as config_file:
        json.dump(config, config_file)

    # Write the command to an sh file
    sh_file_path = os.path.join('/scratch/carmon/lcluc_fire/data/commands.sh')
    with open(sh_file_path, 'a') as sh_file:
        sh_file.write(f"python run_map_traits.py {config_file_path}\n")

print("Processing complete.")
