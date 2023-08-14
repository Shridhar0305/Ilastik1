import subprocess
import os
import sys
import time
from termcolor import colored


def run_pixel_classifier(ilastik_loc, temp_dir, pixel_project, input_files, show_output=True):
    args = []

    args.append(f'{ilastik_loc}\\run-ilastik.bat')
    args.append('--headless')
    args.append(f'--project={pixel_project}')
    args.append('--export_source=Probabilities Stage 2')
    args.append('--output_format=hdf5')
    args.append(f'--output_filename={temp_dir}\\{{nickname}}.hdf5')
    args.extend(input_files)

    kwargs = {}
    kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
    kwargs['shell'] = False

    if not show_output:
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    else:
        proc = subprocess.Popen(args, stdin=subprocess.DEVNULL)

    while True:
        r_code = proc.poll()
        time.sleep(.1)

        if r_code != None:
            return r_code



def run_object_classifier(ilastik_loc, temp_dir, object_project, prediction_map, input_file, show_output=True):
    args = []

    args.append(f'{ilastik_loc}\\run-ilastik.bat')
    args.append('--headless')
    args.append(f'--project={object_project}')
    args.append('--export_source=Object Probabilities')
    args.append(f'--table_filename={temp_dir}\\{{nickname}}.csv')
    args.append('--prediction_maps')
    args.append(prediction_map)
    args.append('--raw_data')
    args.append(input_file)
    
    kwargs = {}
    kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
    kwargs['shell'] = False

    if not show_output:
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    else:
        proc = subprocess.Popen(args, stdin=subprocess.DEVNULL)

    while True:
        r_code = proc.poll()
        time.sleep(.1)

        if r_code != None:
            return r_code



def check_pixel_output_files(temp_dir, nicknames):

    nicknames = [nickname + '.h5' for nickname in nicknames]

    for _file in os.listdir(temp_dir):
        if _file in nicknames:
            nicknames.remove(_file)
    
    return nicknames
