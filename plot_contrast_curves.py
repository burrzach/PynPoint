#imports
import csv
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path
from tqdm import trange
from astropy.stats.sigma_clipping import sigma_clip
from astroquery.simbad import Simbad

#settings
props_file = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/star_data.csv"
psf_obs = '2023-06-15-2'
curves_folder = ...

#functions
def apparent_to_absolute(mag, distance):
    return mag - 5 * np.log10(distance) - 5

def open_evolutionary_tracks(input_dir, extension="txt", rows_header=2):
    '''Read in csv data from evolutionary tracks.

    Opens all csv files in a given directory and reads in the header and data
    seperately. Only the header from the first file is read.

    Parameters
    ----------
    input_dir: str
        The directory where the evolutionary tracks are stored.
    extension: str, default='txt'
        The file extension for all tracks. If there are multiple extensions and
        no other files in the target directory, an asterisk * can be used.
    rows_header: int, default=2
        The amount of rows of the header in each file. 

    Returns
    -------
    header: list
        The header of the files, containing the names and units of every column.
    data_all: list(np.ndarray)
        A list containing arrays of each evolutionary track.
    '''
    file_list = sorted(glob.glob(input_dir + f'*.{extension}'))
    if len(file_list) == 0:
        raise FileNotFoundError(f"No evolutionary tracks found in {input_dir}")
    csv.register_dialect('any_spaces', delimiter=' ', skipinitialspace=True)
    header = []
    data_all = []
    for track in file_list:
        data = []
        with open(track, 'r') as f:
            reader=csv.reader(f , dialect='any_spaces')
            for i, row in enumerate(reader):
                if i < rows_header:
                    if len(header) < rows_header:
                        header.append(list(filter(None,row[1:])))
                else:
                    data.append(row[:-1]) # Remove trailing whitespace (last entry)

        data = np.array(data, dtype=float)
        data_all.append(data)

    return header, data_all

#load star properties
star_props = pd.read_csv(props_file, index_col=0)
fake_app_mag = star_props.loc[star_props['obs'] == psf_obs, 'J'].iloc[0]

#load data
file = ...
contrast_map = np.genfromtxt(file)

seps = int((contrast_map.shape[0] / 2))
sep_space = contrast_map[1:seps, 0]
angle_space = contrast_map[0, 1:]

pre_map = contrast_map[1:seps, 1:]
post_map = contrast_map[seps+1:, 1:]

#grab properties
obs = file[-29:-17]
while obs[0] != '2':
    obs = obs[1:]

dist = star_props.loc[star_props['obs'] == obs, 'dist'].iloc[0]
age = star_props.loc[star_props['obs'] == obs, 'age'].iloc[0]

# #calculations
# sep_space = pre_map[1:, 0]
# angle_space = pre_map[0, 1:]

# pre_map = pre_map[1:, 1:]
# post_map = post_map[1:, 1:]

# pre_curve = [np.mean(pre_map[i]) for i in range(len(pre_map))]
# pre_error = [np.std(pre_map[i]) for i in range(len(pre_map))]
# post_curve = [np.mean(post_map[i]) for i in range(len(post_map))]
# post_error = [np.std(post_map[i]) for i in range(len(post_map))]


# #plot
# plt.figure()
# plt.errorbar(sep_space, pre_curve, yerr=pre_error, marker='o', capsize=3, label='Before SDI')
# plt.errorbar(sep_space, post_curve, yerr=post_error, marker='o', capsize=3, label='After SDI')
# plt.gca().invert_yaxis()
# plt.xlim(xmin=0)
# plt.xlabel('Separation [arcsec]')
# plt.ylabel('Contrast [-]')
# plt.legend()
# plt.show()
