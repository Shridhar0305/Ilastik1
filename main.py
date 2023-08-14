import configparser
import csv
from enum import unique
import math
import re
import os
import sys
import signal
import functools
import time
from numpy.core.shape_base import block

from numpy.lib.shape_base import tile
from Extractor.classifier import *
from Extractor.analyzer import *
from termcolor import colored
from threading import Thread, Event
from queue import Empty, Queue


SOURCEDIR = ""
MAXNUM = 0
VERBOSE = True
PIXEL_PROJ = ""
OBJECT_PROJ = ""
ILASTIK_INSTALL = ""
TEMPDIR = ""
SHOWILASTIKOUT = False
PIXELSPERMM = 1.0
TO_EXCLUDE = SeedExcluder(0.60, 300, 300, 300, 300, 300, 300)
OUTPUT_CSV = ""
NUM_THREADS = 2
DELAYBTWTHREAD = 0


def purge_tmp():
    for file in os.listdir(TEMPDIR):
        if file.endswith('.h5'):
            if VERBOSE:
                print(colored(f'Deleting file in temp: {file}', 'yellow'))
            os.remove(TEMPDIR + '\\' + file)


def purge_src():
    for file in os.listdir(SOURCEDIR):
        if file.endswith('.h5'):
            if VERBOSE:
                print(colored(f'Deleting file in source: {file}', 'yellow'))
            os.remove(SOURCEDIR + '\\' + file)


def extract_plain_fname(path):
    start = path.rfind('\\')
    end = path.rfind('.')
    if end == -1 and start != -1:
        return path[start+1:]
    if end == -1 and start == -1:
        return path
    if end != -1 and start == -1:
        return path[:end]
    if start > end:
        return None
    return path[start+1: end]


def seperate_list_by_len(li, N):
    N = min(len(li), N)
    chunk_sz = math.ceil(len(li) / N)
    return [[li[i] for i in range(N * chunk_n, min(N * (chunk_n + 1), len(li)))] for chunk_n in range(chunk_sz)]


def add_dir_to_files(dirName, files, ext=''):
    # if not dirName.endswith('\\'):
    #     dirName += '\\'
    return [dirName + file + ext for file in files]

def arrange_files_to_nick(predmapfiles, files):
    retfiles = []
    for predmap in predmapfiles:
        for file in files:
            if extract_plain_fname(file) == extract_plain_fname(predmap):
                retfiles.append(file)
                break
    return retfiles


def get_next_avail_fName(pathWithoutExt, ext):
    next_i = 2
    if ext:
        ext = '.' + ext

    while True:
        fName = f'{pathWithoutExt}_{next_i}{ext}'
        if not os.path.isfile(fName):
            return fName
        next_i += 1


def get_unique_filename(path):
    # Return it if that file doesnt exist
    if not os.path.isfile(path):
        return path

    end = path.rfind('.')
    ext = ''
    pathWithoutExt = path

    if end != -1:
        ext = path[end + 1:]
        pathWithoutExt = path[:end]

    return get_next_avail_fName(pathWithoutExt, ext)


def handle_csv_out(csv_q, thrd_done):
    fName = get_unique_filename(OUTPUT_CSV)
    print(colored(f'Writing to file: {fName}', 'blue'))
    csv_fp = open(fName, 'w', newline='')
    writer = csv.writer(csv_fp)
    # writer.writerow(['Sample Name', 'Number Hulled', 'Number Naked', 'Number Spikelet',
    #         'Hulled Average Area', 'Hulled Area Standard Deviation', 'Naked Average Area', 'Naked Area Standard Deviation', 'Spikelet Average Area', 'Spikelet Area Standard Deviation',
    #         'Hulled Average Circularity', 'Hulled Circularity Standard Deviation', 'Naked Average Circularity', 'Naked Circularity Standard Deviation', 'Spikelet Average Circularity', 'Spikelet Circularity Standard Deviation',
    #         'Hulled Average length', 'Hulled length Standard Deviation', 'Naked Average length', 'Naked length Standard Deviation', 'Spikelet Average length','Spikelet length Standard Deviation', 
    #         'Hulled Average width', 'Hulled width Standard Deviation', 'Naked Average width', 'Naked width Standard Deviation', 'Spikelet Average width', 'Spikelet width Standard Deviation'
    #         ])
    writer.writerow(['sample', 'hull_N', 'nkd_N', 'spklt_N', 'hull_F', 'nkd_F', 'spklt_F',
        'hull_A', 'hull_A_SD', 'nkd_A', 'nkd_A_SD', 'spklt_A', 'spklt_A_SD',
        'hull_C', 'hull_C_SD', 'nkd_C', 'nkd_C_SD', 'spklt_C', 'spklt_C_SD',
        'hull_L', 'hull_L_SD', 'nkd_L', 'nkd_L_SD', 'spklt_L', 'spklt_L_SD', 
        'hull_W', 'hull_W_SD', 'nkd_W', 'nkd_W_SD', 'spklt_W', 'spklt_W_SD'
        ])
    while not thrd_done.is_set():
        try:
            row = csv_q.get(block=True, timeout=2)
        except Empty:
            continue
        writer.writerow(row.values())
        csv_fp.flush()
    csv_fp.close()
    print(colored(f'Closing file: {fName}', 'blue'))
    
    input_csv_filename = OUTPUT_CSV  
    output_base_filename = "output"   # Base name for output CSV files

    # Open the input CSV file for reading
    with open(input_csv_filename, "r", newline="") as input_csv_file:
        reader = csv.reader(input_csv_file)
        header = next(reader)  # Read the header row

        # Create a dictionary to store separate output file handlers
        output_files = {}

        for row in reader:
            # Assuming the row contains an identifier in the first column
            identifier = row[0]
            output_filename = f"{output_base_filename}_{identifier}.csv"

            # If the output file for this identifier is not open yet, open it
            if identifier not in output_files:
                output_files[identifier] = open(output_filename, "w", newline="")
                writer = csv.writer(output_files[identifier])
                writer.writerow(header)

            # Write the row to the appropriate output file
            writer.writerow(row)

        # Close all the output files
        for output_file in output_files.values():
            output_file.close()

if __name__ == "__main__":
    main()



def PreProcessData(data):
    if get_idx_for_name(data, 'Predicted Class') == -1:
        return False
    if get_idx_for_name(data, 'Probability of Hulled') == -1:
        return False
    if get_idx_for_name(data, 'Probability of Naked') == -1:
        return False
    if get_idx_for_name(data, 'Probability of Spikelet') == -1:
        return False
    if get_idx_for_name(data, 'Radii of the object_0') == -1:
        return False
    if get_idx_for_name(data, 'Radii of the object_1') == -1:
        return False
    if get_idx_for_name(data, 'Size in pixels') == -1:
        return False

    return True


def AddRejectedRow(nick):
    output = {}
    output['Sample Name'] = nick
    output['Number Hulled'] = output['Number Naked'] = output['Number Spikelet'] = ''
    output['Hulled Filtered'] = output['Naked Filtered'] = output['Spikelet Filtered'] = ''
    output['Hulled Average Area']
    output['Hulled Area Standard Deviation'] = output['Naked Average Area'] = output['Naked Area Standard Deviation'] = output['Spikelet Average Area'] = output['Spikelet Area Standard Deviation'] = ''

    output['Hulled Average Circularity'] = output['Hulled Circularity Standard Deviation'] = output['Naked Average Circularity'] = output['Naked Circularity Standard Deviation'] = output['Spikelet Average Circularity']  = output['Spikelet Circularity Standard Deviation'] = ''
    output['Hulled Average length'] = output['Hulled length Standard Deviation'] = output['Naked Average length'] = output['Naked length Standard Deviation'] = output['Spikelet Average length'] = output['Spikelet length Standard Deviation'] = ''
    output['Hulled Average width'] = output['Hulled width Standard Deviation'] = output['Naked Average width'] = output['Naked width Standard Deviation'] = output['Spikelet Average width'] = output['Spikelet width Standard Deviation'] = ''

    return output


def ProcessCSVData(nick, csvdata):
    output = {}
    output['Sample Name'] = nick

    dic = apply_threshold_to_idxs(get_idxs_for_type_seed(csvdata), csvdata, TO_EXCLUDE, PIXELSPERMM)

    unfiltered = get_number_for_type_seed(csvdata)
    output['Number Hulled'] = unfiltered['Hulled']
    output['Number Naked'] = unfiltered['Naked']
    output['Number Spikelet'] = unfiltered['Spikelet']

    passed = dic.copy()
    for x in passed.keys():
        passed[x] = unfiltered[x] - len(passed[x])
    removed = passed

    output['Hulled Filtered'] = removed['Hulled']
    output['Naked Filtered'] = removed['Naked']
    output['Spikelet Filtered'] = removed['Spikelet']


    tmp = get_area_for_type_seed(csvdata, dic.copy(), PIXELSPERMM)
    tmp = do_stats_on_dic(tmp)
    output['Hulled Average Area'] = tmp['Hulled']['Avg']
    output['Hulled Area Standard Deviation'] = tmp['Hulled']['Std']
    output['Naked Average Area'] = tmp['Naked']['Avg']
    output['Naked Area Standard Deviation'] = tmp['Naked']['Std']
    output['Spikelet Average Area'] = tmp['Spikelet']['Avg']
    output['Spikelet Area Standard Deviation'] = tmp['Spikelet']['Std']

    tmp = get_circ_for_types(csvdata, dic.copy())
    tmp = do_stats_on_dic(tmp)
    output['Hulled Average Circularity'] = tmp['Hulled']['Avg']
    output['Hulled Circularity Standard Deviation'] = tmp['Hulled']['Std']
    output['Naked Average Circularity'] = tmp['Naked']['Avg']
    output['Naked Circularity Standard Deviation'] = tmp['Naked']['Std']
    output['Spikelet Average Circularity'] = tmp['Spikelet']['Avg']
    output['Spikelet Circularity Standard Deviation'] = tmp['Spikelet']['Std']

    min_feret, max_feret = get_min_and_max_feret(csvdata, dic.copy(), PIXELSPERMM)
    min_feret = do_stats_on_dic(min_feret)
    max_feret = do_stats_on_dic(max_feret)
    output['Hulled Average length'] = max_feret['Hulled']['Avg']
    output['Hulled length Standard Deviation'] = max_feret['Hulled']['Std']
    output['Naked Average length'] = max_feret['Naked']['Avg']
    output['Naked length Standard Deviation'] = max_feret['Naked']['Std']
    output['Spikelet Average length'] = max_feret['Spikelet']['Avg']
    output['Spikelet length Standard Deviation'] = max_feret['Spikelet']['Std']

    output['Hulled Average width'] = min_feret['Hulled']['Avg']
    output['Hulled width Standard Deviation'] = min_feret['Hulled']['Std']
    output['Naked Average width'] = min_feret['Naked']['Avg']
    output['Naked width Standard Deviation'] = min_feret['Naked']['Std']
    output['Spikelet Average width'] = min_feret['Spikelet']['Avg']
    output['Spikelet width Standard Deviation'] = min_feret['Spikelet']['Std']

    return output


def BatchWork(csv_q, work_q, thrd_id):
    temp_dir = TEMPDIR
    if not temp_dir.endswith('\\'):
        temp_dir += '\\'

    time.sleep((thrd_id - 1) * DELAYBTWTHREAD)

    if VERBOSE:
        print(colored(f'Thread {thrd_id} has started', 'blue'))

    while True:
        try:
            batch = work_q.get(block=True, timeout=2)
        except Empty:
            if VERBOSE:
                print(colored(f'Thread {thrd_id} has finished', 'blue'))
            break

        nicknames = [extract_plain_fname(file) for file in batch]

        if VERBOSE:
            st_files = 'INFO: Running pixel classification with files: ' + ', '.join(batch)
            print(colored(st_files, 'blue'))

        run_pixel_classifier(ILASTIK_INSTALL, TEMPDIR, PIXEL_PROJ, batch, SHOWILASTIKOUT)

        notfound = check_pixel_output_files(TEMPDIR, nicknames)
        if notfound:
            filesNotFound = ', '.join(notfound)
            print(colored(f'ERROR: Pixel Classifier failed to output the following files: {filesNotFound}', 'red'))
            for unfound in notfound:
                nicknames.remove(extract_plain_fname(unfound))

        # We have to seperately run the object classifier
        predmapfiles = add_dir_to_files(temp_dir, nicknames, '.h5')
        files = arrange_files_to_nick(predmapfiles, batch)
        assert(len(files) == len(predmapfiles))

        for i, predmap in enumerate(predmapfiles):
            nick = extract_plain_fname(predmap)

            if VERBOSE:
                st_file = 'INFO: Running object classification for file: ' + files[i]
                print(colored(st_file, 'blue'))

            run_object_classifier(ILASTIK_INSTALL, TEMPDIR, OBJECT_PROJ, predmap, batch[i], SHOWILASTIKOUT)
            # NOTE: For some reason it keeps adding _table to the csv name
            table = temp_dir + nick + '_table.csv'
            if not os.path.isfile(table):
                print(colored(f'ERROR: Object Classifier failed to output {nick}.csv', 'red'))
                continue
            csvdata = read_csv_as_np(table)
            if len(csvdata) == 0:
                print(colored(f'ERROR: No data found in {nick}.csv', 'red'))
                continue
            if PreProcessData(csvdata):
                processed_data = ProcessCSVData(nick, csvdata)
            else:
                print(colored(f'ERROR: Incomplete data found in {nick}.csv, adding empty entry to csv', 'red'))
                processed_data = AddRejectedRow(nick)
            #print(colored(processed_data, 'yellow'))
            csv_q.put(processed_data)



def StartFileProcessing(csv_q, ToProcess):
    # Make sure nothing can stop until this file is processed
    proc_thrds = []
    
    work_q = Queue()
    for files in ToProcess: # files is a list of files
        work_q.put(files)

    for i in range(NUM_THREADS):
        n_trd = Thread(target=BatchWork, args=(csv_q, work_q, i+1))
        n_trd.start()
        proc_thrds.append(n_trd)
    
    # Wait for threads to complete
    for thrd in proc_thrds:
        thrd.join()


def main():
    global SOURCEDIR
    global MAXNUM
    global VERBOSE
    global PIXEL_PROJ
    global OBJECT_PROJ
    global ILASTIK_INSTALL
    global TEMPDIR
    global SHOWILASTIKOUT
    global PIXELSPERMM
    global TO_EXCLUDE
    global OUTPUT_CSV
    global NUM_THREADS
    global DELAYBTWTHREAD

    os.system('color')


    config = configparser.ConfigParser()
    if not config.read('config.ini'):
        print('Cannot read config.ini')
        exit(2)

    try:
        SOURCEDIR = config.get('SETTINGS', 'SourceFolder')
        PATTERN = config.get('SETTINGS', 'Pattern')
        CASEINSENSITIVE = config.getboolean('SETTINGS', 'CaseInsensitivePattern')
        MAXNUM = config.getint('SETTINGS', 'MaxNum')
        VERBOSE = config.getboolean('SETTINGS', 'ShowProgramProgress')

        PIXEL_PROJ = config.get('SETTINGS', 'PixelClassificationProject')
        OBJECT_PROJ = config.get('SETTINGS', 'ObjectClassificationProject')
        ILASTIK_INSTALL = config.get('SETTINGS', 'IlastikInstallationFolder')
        SHOWILASTIKOUT = config.getboolean('SETTINGS', 'ShowIlastikOutput')

        DELAYBTWTHREAD = config.getfloat('SETTINGS', 'DelayBetweenThrdStartup')

        TEMPDIR = config.get('SETTINGS', 'TemporaryFolder')

        PIXELSPERMM = config.getfloat('SETTINGS', 'PixelsPerMM')
        threshold = config.getfloat('SETTINGS', 'RejectUnder')

        maxHulledLength = config.getfloat('SETTINGS', 'MaxHulledLength')
        maxHulledWidth = config.getfloat('SETTINGS', 'MaxHulledWidth')
        maxNakedLength = config.getfloat('SETTINGS', 'MaxNakedLength')
        maxNakedWidth = config.getfloat('SETTINGS', 'MaxNakedWidth')
        maxSpikeletLength = config.getfloat('SETTINGS', 'MaxSpikeletLength')
        maxSpikeletWidth = config.getfloat('SETTINGS', 'MaxSpikeletWidth')

        TO_EXCLUDE = SeedExcluder(threshold, maxHulledWidth, maxHulledLength, maxNakedWidth, maxNakedLength, maxSpikeletWidth, maxSpikeletLength)

        OUTPUT_CSV = config.get('SETTINGS', 'OutputCSVFile')

        NUM_THREADS = config.getint('SETTINGS', 'NumThreads')

        RemoveUnwantedInTemp = config.getboolean('SETTINGS', 'RemoveUnwantedInTemp')
        RemoveUnwantedInSrc = config.getboolean('SETTINGS', 'RemoveUnwantedInSrc')

    except Exception as e:
        print(colored(f'Cannot read from config.ini: {e}', 'red'))
        exit(2)

    if CASEINSENSITIVE:
        reg = re.compile(PATTERN, re.IGNORECASE)
    else:
        reg = re.compile(PATTERN)

    if not SOURCEDIR.endswith('\\'):
        SOURCEDIR += '\\'

    filesToProcess = []
    for _file in os.listdir(SOURCEDIR):
        _path = SOURCEDIR + _file
        if reg.match(_file) and os.path.isfile(_path):
            filesToProcess.append(_path)
    
    if len(filesToProcess) == 0:
        print(colored(f'WARNING: No file to process in: {SOURCEDIR}', 'yellow'))
        exit(1)

    ToProcess = seperate_list_by_len(filesToProcess, MAXNUM)

    if VERBOSE:
        layout =  'INFO: Processing into chunks of size [' + ', '.join(list(map(lambda x: str(len(x)), ToProcess))) + ']'
        print(colored(layout, 'blue'))
    
    csv_q = Queue()
    thrd_done = Event()
    csv_thrd = Thread(target=handle_csv_out, args=(csv_q, thrd_done))
    csv_thrd.start()

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    StartFileProcessing(csv_q, ToProcess)
    thrd_done.set()
    csv_thrd.join()

    if RemoveUnwantedInTemp:
        purge_tmp()
    if RemoveUnwantedInSrc:
        purge_src()


if __name__ == "__main__":
    main()
