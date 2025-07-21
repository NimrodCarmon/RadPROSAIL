import json
import os
import glob
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import pdb

def process_file(csv_file, output_dir, log_file):
    if csv_file.endswith(('_chan.csv', '_flux.csv', '_scan.csv')):
        return

    with open(csv_file, 'r') as file:
        lines = file.readlines()
        case_count = 0
        is_case_open = False
        for line in lines:
            if re.match(r'case index \d+ = \{', line.strip()):
                is_case_open = True
            elif line.strip() == '}' and is_case_open:
                case_count += 1
                is_case_open = False

    if case_count < 9:
        base_name = os.path.splitext(csv_file)[0]
        with open(log_file, 'a') as log:
            log.write(csv_file + '\n')
        for ext in ['.csv', '_chan.csv', '_flux.csv', '_scan.csv']:
            file_to_delete = base_name + ext
            if os.path.exists(file_to_delete):
                # os.remove(file_to_delete)
                print(f"Deleting {file_to_delete}")

def process_files_v2(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    output_dir = data['output']['output_dir'] + "/output"

    csv_files = [f for f in glob.glob(os.path.join(output_dir, '*.csv')) if not f.endswith(('_chan.csv', '_flux.csv', '_scan.csv'))]

    log_file = 'bad_files_log.txt'

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, csv_file, output_dir, log_file) for csv_file in csv_files]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            pass

    return "File processing completed."

def process_files_from_list(filelist):
    with open(filelist, 'r') as file:
        csv_files = [line.strip() for line in file.readlines()]

    pdb.set_trace()
    all_files_valid = True

    for csv_file in csv_files:
        with open(csv_file, 'r') as file:
            lines = file.readlines()
            case_count = 0
            is_case_open = False
            for line in lines:
                if re.match(r'case index \d+ = \{', line.strip()):
                    is_case_open = True
                elif line.strip() == '}' and is_case_open:
                    case_count += 1
                    is_case_open = False

        if case_count < 9:
            print(f"Invalid file (less than 9 cases): {csv_file}")
            all_files_valid = False
        else:
            print(f"Valid file: {csv_file}")

    if all_files_valid:
        print("All files are valid.")
    else:
        print("Some files are invalid.")

    return

def main():

    runtype = int(sys.argv[1]) # one for running over dir, 2 for running over file
    pdb.set_trace()
    if runtype==1:
        process_files_v2('/scratch/carmon/modtran_luts/projects/universal_9case/lut_config2.json') 
    elif runtype==2:
        filelist = sys.argv[2]
        process_files_from_list(filelist)

if __name__ == "__main__":
    main()
