import os
import subprocess
import time

class ClassificationPipeline(object):

    """
    This class holds the main pipeline for the extractor

    @ilastic_loc: location of ilastik installation
    @temp_dir: prefferd directory to store intermediate files from pixel classifier and object prediction

    """
    
    def __init__(self, ilastik_loc, temp_dir=".\\tmp\\", show_output=True):
        self.ilastik_loc = ilastik_loc
        self.temp_dir = temp_dir
        self.show_output = show_output
        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)

    '''
    @nicknames: a list of the nicknames of files (defined by ilastik) 
    
    returns files that were not found
    '''
    def check_pixel_output_files(self, nicknames):

        nicknames = [nickname + '.h5' for nickname in nicknames]

        for _file in os.listdir(self.temp_dir):
            if _file in nicknames:
                nicknames.remove(_file)
        
        return nicknames

        
    '''
    @pixel_project: location of the pixel project
    @input_files: a list of filenames to be processed
    '''
    def run_pixel_classifier(self, pixel_project, input_files):
        args = []

        args.append(f'{self.ilastik_loc}\\run-ilastik.bat')
        args.append('--headless')
        args.append(f'--project={pixel_project}')
        args.append('--export_source=Probabilities Stage 2')
        args.append('--output_format=hdf5')
        args.append(f'--output_filename={self.temp_dir}\\{{nickname}}.hdf5')
        args.extend(input_files)

        if not self.show_output:
            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            proc = subprocess.Popen(args)

        while True:
            r_code = proc.poll()
            time.sleep(.1)
            if r_code != None:
                return r_code

    '''
    @object_project: location of the object project
    @prediction_map: prediction_map outputed by pixel classifier (single item)
    @input_file: the same input that you gave to pixel classifier (single item)
    @nickname: nickname as defined by ilastik (no exts)
    '''
    def run_object_classifier(self, object_project, prediction_map, input_file, nickname):
        args = []

        args.append(f'{self.ilastik_loc}\\run-ilastik.bat')
        args.append('--headless')
        args.append(f'--project={object_project}')
        args.append('--export_source=Object Probabilities')
        args.append(f'--table_filename={self.temp_dir}\\{{nickname}}.csv')
        args.append('--prediction_maps')
        args.append(prediction_map)
        args.append('--raw_data')
        args.append(input_file)

        if not self.show_output:
            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            proc = subprocess.Popen(args)

        while True:

            r_code = proc.poll()
            time.sleep(.1)
            if r_code != None:
                return r_code


class OldSeedExcluder(object):
    def __init__(self, threshold, hulledMaxWid, hulledMaxLen, nakedMaxWid, nakedMaxLen, spikeletMaxWid, spikeletMaxLen):
        self.threshold = threshold
        self.hulledMaxWid = hulledMaxWid
        self.hulledMaxLen = hulledMaxLen
        self.nakedMaxWid = nakedMaxWid
        self.nakedMaxLen = nakedMaxLen
        self.spikeletMaxWid = spikeletMaxWid
        self.spikeletMaxLen = spikeletMaxLen
    
    def check_thrshold(self, thrs):
        return thrs < self.threshold

    def check_hullmaxwid(self, hullmaxwid):
        return hullmaxwid < self.hulledMaxWid

    def check_hullmaxlen(self, hullmaxlen):
        return hullmaxlen < self.hulledMaxLen

    def check_nnkdmaxlen(self, nkdmaxlen):
        return nkdmaxlen < self.nakedMaxLen

    def check_nnkdmaxwid(self, nkdmaxwid):
        return nkdmaxwid < self.nakedMaxWid

    def check_spkltmaxwid(self, spkltmaxwid):
        return spkltmaxwid < self.spikeletMaxWid

    def check_spkltmaxlen(self, spkltmaxlen):
        return spkltmaxlen < self.spikeletMaxLen