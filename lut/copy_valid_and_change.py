import os
import re
import argparse
import concurrent.futures
from tqdm import tqdm

def process_file(filename):
    original_params = {"AOT550": "", "GNDALT": "", "H2OSTR": "", "SZA": ""}
    modified_params = original_params.copy()

    with open(os.path.join(dir_path, filename), 'r') as file:
        contents = file.read()

        # Extract initial parameter values from filename
        for param in original_params.keys():
            match = re.search(fr"{param}-(-?[\d.]+)", filename)
            if match:
                original_params[param] = match.group(1)
                modified_params[param] = match.group(1)

        # Check for warnings and update parameters accordingly
        aod_warnings = re.findall(r"Warning from routine GETVIS:.*AOD =.*is (less than|more than).*AOD \(= (.*?)\)", contents, re.DOTALL)
        h2o_warnings = re.findall(r"Warning from routine SCLCOL:.*Input water column,.*is (above maximum allowed|below minimum allowed).*maximum, (.*?) gm/cm2", contents, re.DOTALL)

        if aod_warnings:
            modified_params["AOT550"] = "{:.5f}".format(float(aod_warnings[0][1].strip()))
        if h2o_warnings:
            modified_params["H2OSTR"] = "{:.5f}".format(float(h2o_warnings[0][1].strip()))

        # Log to output file if any parameter was modified
        if original_params != modified_params:
            with open(output_file_path, 'a') as output_file:
                output_file.write(f"{filename},{original_params['AOT550']},{original_params['H2OSTR']},{modified_params['AOT550']},{modified_params['H2OSTR']}\n")

# Set up argument parser
parser = argparse.ArgumentParser(description='Process .tp6 files for AOD and H2O warnings and log parameter changes.')
parser.add_argument('dir_path', type=str, help='Directory to scan for .tp6 files')
args = parser.parse_args()

# Define the directory path and output file path
dir_path = args.dir_path
output_file_path = os.path.join(dir_path, 'parameter_changes.csv')

# Initialize the output file with headers
with open(output_file_path, 'w') as output_file:
    output_file.write("filename,original_AOT550,original_H2OSTR,modified_AOT550,modified_H2OSTR\n")

# Get all the .tp6 files
tp6_files = [f for f in os.listdir(dir_path) if f.endswith(".tp6")]
total_files = len(tp6_files)

# Use a ProcessPoolExecutor to process the files in parallel
with concurrent.futures.ProcessPoolExecutor() as executor:
    list(tqdm(executor.map(process_file, tp6_files), total=total_files))

print("Processing and logging of parameter changes completed!")
