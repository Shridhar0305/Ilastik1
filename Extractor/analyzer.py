import csv
import numpy as np
import math

from termcolor import colored

class InvalidDataValue(Exception):
    pass

class SeedExcluder(object):
    
    def __init__(self, threshold, hulledMaxWid, hulledMaxLen, nakedMaxWid, nakedMaxLen, spikeletMaxWid, spikeletMaxLen):
        self.threshold = threshold
        self.hulledMaxWid = hulledMaxWid
        self.hulledMaxLen = hulledMaxLen
        self.nakedMaxWid = nakedMaxWid
        self.nakedMaxLen = nakedMaxLen
        self.spikeletMaxWid = spikeletMaxWid
        self.spikeletMaxLen = spikeletMaxLen


def read_csv_as_np(fName):
    try:
        with open(fName, 'r') as dest_f:
            data_iter = csv.reader(dest_f, delimiter=',', quotechar='"')
            data = [data for data in data_iter]
        return np.asarray(data)
    except FileNotFoundError:
        return []


def write_single_row_csv_from_dict(dic, fName):
    with open(fName, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(dic.keys())
        writer.writerow([dic[x] for x in dic.keys()])


def get_idx_for_name(data, name):
    for i, title in enumerate(data[0]):
        if title == name:
            return i
    return -1


def get_col_by_name(data, name):
    col = get_idx_for_name(data, name)
    if col == -1:
        # we failed to find the appropriate column
        return None
    return data[1:,col]


def get_idxs_for_type_seed(data):
    coldata = list(get_col_by_name(data, 'Predicted Class'))
    dic = {}
    for x in ['Hulled', 'Naked', 'Spikelet']:      
        dic[x] = [i for i, row in enumerate(coldata) if row == x]
    return dic


def get_number_for_type_seed(data):
    dic = get_idxs_for_type_seed(data)
    for x in dic.keys():
        dic[x] = len(dic[x])
    return dic


def apply_threshold_to_idxs(dic, data, exclude, pixelspermm):
    p_hulled = get_col_by_name(data, 'Probability of Hulled')
    p_naked = get_col_by_name(data, 'Probability of Naked')
    p_spiklt = get_col_by_name(data, 'Probability of Spikelet')
    
    list_max = map(lambda x, y, z: max(float(x), float(y), float(z)), p_hulled, p_naked, p_spiklt)

    after_threshold =  [i for i, x in enumerate(list_max) if x < exclude.threshold]

    radii1 = get_col_by_name(data, 'Radii of the object_0')
    radii2 = get_col_by_name(data, 'Radii of the object_1')

    length = [max(float(r1), float(r2))*4/pixelspermm for r1, r2 in zip(radii1, radii2)]
    width = [min(float(r1), float(r2))*4/pixelspermm for r1, r2 in zip(radii1, radii2)]

    new_hulled = dic['Hulled'].copy()

    for idx in dic['Hulled']:
        if idx in after_threshold or length[idx] > exclude.hulledMaxLen or width[idx] > exclude.hulledMaxWid:
            new_hulled.remove(idx)
    
    dic['Hulled'] = new_hulled

    new_naked = dic['Naked'].copy()
    
    for idx in dic['Naked']:
        if idx in after_threshold or length[idx] > exclude.nakedMaxLen or width[idx] > exclude.nakedMaxWid:
            new_naked.remove(idx)
    
    dic['Naked'] = new_naked

    new_spiklt = dic['Spikelet'].copy()
    
    for idx in dic['Spikelet']:
        if idx in after_threshold or length[idx] > exclude.spikeletMaxLen or width[idx] > exclude.spikeletMaxWid:
            new_spiklt.remove(idx)
    
    dic['Spikelet'] = new_spiklt

    return dic


def get_area_for_type_seed(data, dic, pixelspermm):
    coldata = get_col_by_name(data, 'Size in pixels')
    for x in dic.keys():
        dic[x] = [float(coldata[idx])/pixelspermm for idx in dic[x]]
    return dic


def get_min_and_max_feret(data, dic, pixelspermm):
    col1 = get_col_by_name(data, 'Radii of the object_0')
    col2 = get_col_by_name(data, 'Radii of the object_1')

    max_feret = [max(float(r1), float(r2)) for r1, r2 in zip(col1, col2)]
    min_feret = [min(float(r1), float(r2)) for r1, r2 in zip(col1, col2)]
    max_dic = {}
    min_dic = {}
    for x in dic.keys():
        max_dic[x] = [max_feret[idx]*4/pixelspermm for idx in dic[x]]
        min_dic[x] = [min_feret[idx]*4/pixelspermm for idx in dic[x]]

    return (min_dic, max_dic)


def get_circ_for_types(data, dic):
    a = get_col_by_name(data, 'Radii of the object_0')
    b = get_col_by_name(data, 'Radii of the object_1')
    for x in dic.keys():
        #dic[x] = [get_circularity(float(a[idx]), float(b[idx])) for idx in dic[x]]
        li = []
        for idx in dic[x]:
            try:
                li.append(get_circularity(float(a[idx]), float(b[idx])))
            except InvalidDataValue:
                print('Here')
                continue
        dic[x] = li
    return dic


def do_stats_on_dic(dic):
    for x in dic.keys():
        dataToAnal = dic[x]
        dic[x] = {
            'Std': get_std(dataToAnal),
            'Avg': get_avg(dataToAnal)
        }
    return dic


def get_std(data):
    try:
        return np.std(data, ddof=1)
    except:
        print('Here')
        return 'nan'


def get_avg(data):
    try:
        return np.mean(data)
    except:
        print('Here')
        return 'nan'


'''
a and b has to be radii
'''
def get_circularity(a, b):
    den = a*a + b*b
    if den == 0:
        raise InvalidDataValue
    return (2*a*b)/den
