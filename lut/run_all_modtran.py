import subprocess
import glob
import os, sys
import time
import argparse
import multiprocessing
import pdb

def main():
    # implement a keyword so this code knows if we are doing csv or channel
    #pdb.set_trace()
    parser = argparse.ArgumentParser()
    parser.add_argument('lut_dir',type=str)
    parser.add_argument('-n_cpu',type=int, default=-1)
    parser.add_argument('-modtran_exe',type=str,default='/Users/carmon/MODTRAN6.0/MODTRAN6.0/bin/macos/mod6c_cons')
    args = parser.parse_args()

    subprocess.call('rm -f /dev/shm/*',shell=True)
    #subprocess.call('cp run_one_modtran.csh {}'.format(args.lut_dir),shell=True)
    os.chdir(args.lut_dir)
    subprocess.call('mkdir logs',shell=True)
    run_files = glob.glob('./*.json', recursive=True)
    lock_files = [os.path.splitext(x)[0] + '.lock' for x in run_files]
    tp6_files = [os.path.splitext(x)[0] + '.tp6' for x in run_files]
    chn_files = [os.path.splitext(x)[0] + '.chn' for x in run_files]

    if (args.n_cpu == -1):
        args.n_cpu = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(processes=args.n_cpu)
    results = []
    for _f in range(len(run_files)):
        results.append(pool.apply_async(run_single_modtran, args=(args.modtran_exe,run_files[_f],[tp6_files[_f],chn_files[_f]],lock_files[_f],)))

    #results = [p.get() for p in results]
    pool.close()
    pool.join()


def run_single_modtran(modtran_exe,json_file,check_files,lock_file):
    #pdb.set_trace()
    

    completed = True
    if type(check_files) is str:
        if (os.path.isfile(check_files) is False):
            completed = False
    else:
        for _f in check_files:
            if (os.path.isfile(_f) is False):
                completed = False
    if completed:
        print('already completed: {}'.format(json_file))
        return

    underway = os.path.isfile(lock_file)
    if underway:
        print('already underway: {}'.format(lock_file))
        return

    log_file_path = '/scratch/carmon/modtran_luts/modtran_lut_builder/log/modtran_processing_time.log'  # Path to the log file
    print(f'starting: {json_file}')
    start_time = time.time()  # Capture start time
    try:
        cmd_str = 'touch {}'.format(lock_file)
        subprocess.call(cmd_str,shell=True)
        cmd_str = '{} {}'.format(modtran_exe, json_file)
        subprocess.call(cmd_str,shell=True)
        cmd_str = 'rm {}'.format(lock_file)
        subprocess.call(cmd_str,shell=True)
    except:
        pass

    end_time = time.time()  # Capture end time

    # Calculate and log the processing time
    processing_time_minutes = (end_time - start_time) / 60
    with open(log_file_path, 'a') as log_file:
        log_file.write(f'{json_file}: {processing_time_minutes:.2f} minutes\n')

    return



def run_modtran_on_dir(lut_dir, n_cpu, modtran_exe, ftype='chn'):
    
    # This function is used to actually run modtran on valid json files
    # valid json files are files that are not being processed RN and that have not been processed already.
    # This here is modified from Phil B's code - thanks Phil.
    # This here is a specific problem with emitsds and must be done
    #pdb.set_trace()
    max_runs = 160 # we will break this after one go with the node

    subprocess.call('rm -f /dev/shm/*',shell=True)
    #subprocess.call('cp run_one_modtran.csh {}'.format(args.lut_dir),shell=True)
    original_dir = os.getcwd()
    os.chdir(lut_dir)
    try:
        subprocess.call('mkdir logs',shell=True)
    except:
        pass
    #pdb.set_trace()
    json_files = glob.glob('./*.json', recursive=True)

    # Filtering out JSON files if a CSV file with the same name exists
    #run_files = [f for f in json_files if not os.path.exists(f"{os.path.splitext(f)[0]}.csv")]
    run_files = [f for f in json_files if not os.path.exists(f"{os.path.splitext(f)[0]}.csv") and not os.path.exists(f"{os.path.splitext(f)[0]}.lock")]
    lock_files = [os.path.splitext(x)[0] + '.lock' for x in run_files]
    # I didn't notice any .lock files created during the runs
    if ftype=='chn':
        tp6_files = [os.path.splitext(x)[0] + '.tp6' for x in run_files]
        chn_files = [os.path.splitext(x)[0] + '.chn' for x in run_files]
        check_files = list(zip(tp6_files, chn_files))
    elif ftype=='csv':
        csv_files = [os.path.splitext(x)[0] + '.csv' for x in run_files]
        check_files = csv_files
    # because the csv file is the last in the list we are safe because the order is important

    debug = False
    if debug is True:
        results = []
        for _f in range(len(run_files)):
            args=(modtran_exe,run_files[_f], check_files ,lock_files[_f],)
            pdb.set_trace()
            temp_result = run_single_modtran(modtran_exe, run_files[_f], check_files[_f] ,lock_files[_f])
            results.append(temp_result)
    else:
        if (n_cpu == -1):
            n_cpu = multiprocessing.cpu_count()

        pool = multiprocessing.Pool(processes=n_cpu)
        results = []
        run = 0
        for _f in range(len(run_files)):
            results.append(pool.apply_async(run_single_modtran, args=(modtran_exe,run_files[_f], check_files[_f] ,lock_files[_f],)))

            run += 1

            if run > max_runs:
                break
        #results = [p.get() for p in results]
        pool.close()
        pool.join()

    os.chdir(original_dir)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()