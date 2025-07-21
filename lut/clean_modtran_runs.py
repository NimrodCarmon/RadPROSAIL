import os
import re
import json
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Process .tp6 files for AOD and H2O warnings.')
parser.add_argument('json_config', type=str, help='Path to the JSON configuration file')
args = parser.parse_args()

# Load the JSON file
with open(args.json_config) as json_file:
    data = json.load(json_file)

# Define the directory path
dir_path = os.path.join(data['output']['output_dir'], 'output')

# Define the output file
output_file = open("output.txt", "w")

# Get all the .tp6 files
tp6_files = [f for f in os.listdir(dir_path) if f.endswith(".tp6")]
total_files = len(tp6_files)

# Loop through all .tp6 files in the directory
for i, filename in enumerate(tp6_files, start=1):
    print(f"Processing file {i} of {total_files}...")
    with open(os.path.join(dir_path, filename), 'r') as file:
        contents = file.read()

        # Check for warnings related to AOD and H2O
        aod_warnings = re.findall(r"Warning from routine GETVIS:.*AOD =.*is (less than|more than).*AOD \(= (.*?)\)", contents, re.DOTALL)
        h2o_warnings = re.findall(r"Warning from routine SCLCOL:.*Input water column,.*is (above maximum allowed|below minimum allowed).*maximum, (.*?) gm/cm2", contents, re.DOTALL)

        # If there are any warnings, write to the output file and print the new values
        if aod_warnings or h2o_warnings:
            output_file.write(f"{filename}:\n")
                
            if aod_warnings:
                for warning in aod_warnings:
                    output_file.write(f"AOD warning: {warning[0]}\n")
                    output_file.write(f"Assigned AOD value: {warning[1]}\n")

            if h2o_warnings:
                for warning in h2o_warnings:
                    output_file.write(f"H2O warning: {warning[0]}\n")
                    output_file.write(f"Assigned H2O value: {warning[1]}\n")

            output_file.write("\n")

# Close the output file
output_file.close()

print("Processing completed!")
