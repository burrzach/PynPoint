#!/usr/bin/env python3

'''
Author: Rick Dullaart

Functions to calculate sensitivity and plot the contrast curve.
'''

import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path
from tqdm import trange
from astropy.stats.sigma_clipping import sigma_clip
from astroquery.simbad import Simbad

PIX_SCALE = 0.01225  # arcsec/pix
M_SUN_TO_M_JUP = 1047.34


def r_theta(im, xc, yc):
    '''Generate a radial mask for an image at given center coordinates.'''
    ny, nx = im.shape
    yp,xp = np.mgrid[0:ny,0:nx]
    yp = yp-yc
    xp = xp - xc
    rr = np.sqrt(yp*yp + xp*xp)
    #phi = np.arctan2(yp, xp)
    return rr#, phi


def get_star_age(target, targetlist="results/targets.csv", skiprows=1):
    '''Retrieve the star age from a csv file.'''
    with open(targetlist, "r", newline="") as csvfile:
        csventries = csv.reader(csvfile, delimiter=",")
        for _ in range(skiprows):
            csventries.__next__()
        for row in csventries:
            if row[0][0] == "#":
                row[0] = row[0][1:]
            if target == row[0] or target == row[1]:
                return int(row[2])
    raise IndexError(f"{target} not found in Target list.")


def query_star_data(identifier, interactive=False):
    '''Query distance and H-band magnitude using Simbad.'''
    if identifier[:2] == 'V_':
        identifier = identifier[2:]
    customSimbad = Simbad()
    customSimbad.add_votable_fields('distance', 'flux(H)')
    result_table = customSimbad.query_object(identifier)
    if len(result_table) == 0:
        if interactive: 
            new_id = input(f"No star found for {identifier}, "
                           + "what name should I query? ")
            result_table = customSimbad.query_object(new_id)
        else:
            raise LookupError(f"Simbad couldn't resolve {identifier}.")
    stardist = result_table['Distance_distance'][0]
    starmag = result_table['FLUX_H'][0]
    return stardist, starmag


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


def mag_to_mass(magnitudes, age_system, track_dir="doc/SPHERE_IRDIS_vega/"):
    '''Convert magnitudes to jupiter masses, given the age of the system.

    Parameters
    ----------
    magnitudes: array_like
        The known contrast magnitudes.
    age_system: float
        The age of the star/planetary system in Myr
    track_dir: str, default="doc/SPHERE_IRDIS_vega/"
        The directory where the evolutionary tracks are stored.

    Returns
    -------
    masses: np.ndarray
        The masses corresponding to the given magnitudes, same shape as x.
    '''
    mag_to_jupmass = []
    _, evolutionary_tracks = open_evolutionary_tracks(track_dir)

    # Magic numbers used: ind 0=mass (M_sun), 1=age (Gyr), 16=magnitude

    for model in evolutionary_tracks:
        # Model age is in Gyr, age_system is in Myr
        ind = np.searchsorted(1000*model[:,1], age_system)
        if ind == len(model):
            ind -= 1
        elif (ind+1 < len(model)) and ((model[ind+1,1] - age_system) < (age_system - model[ind,1])):
            ind+=1  # Check if the value to the right is closer
        if not model[ind,16] == 0:
            mag_to_jupmass.append([model[ind,16], model[ind,0]*M_SUN_TO_M_JUP])

    mag_to_jupmass = np.array(mag_to_jupmass)
    mag_to_jupmass[np.argsort(mag_to_jupmass[:,1])]
    mag_to_jupmass = mag_to_jupmass[::-1] # np.interp needs a monotonically increasing array
    masses = np.interp(magnitudes, mag_to_jupmass[:,0], mag_to_jupmass[:,1])
    return masses


def contrast_rings(data, fluxratio, age, distance, magnitude, sigma=[1,3,5], limits=(30,440)):
    '''Calculate the contrast at radial annuli around the star.'''
    r_img = r_theta(data, data.shape[0]//2, data.shape[1]//2)
    radii = np.array(range(limits[0], limits[1]+1), dtype=np.int16)
    bg, rms = np.zeros(radii.shape), np.zeros(radii.shape)

    for radius in radii[1:-1]:
        mask = (r_img>radius-2) & (r_img<radius+2)
        masked_data = sigma_clip(data[mask])
        mask[mask] = ~masked_data.mask
        bg[radius-limits[0]] = np.mean(data[mask])
        rms[radius-limits[0]] = np.sqrt(np.var(data[mask]))
    bg[bg < 0] = 0
    rms[0], rms[-1] = fluxratio, fluxratio
    mass_limits = []
    for sig in sigma:
        bgmag = -2.5*np.log10((bg+sig*rms) / (fluxratio*PIX_SCALE**2)) + magnitude
        minimum_mass = mag_to_mass(bgmag, age)
        minimum_mass[0], minimum_mass[-1] = 1000, 1000
        mass_limits.append(minimum_mass)
    return radii, mass_limits


def rms_contrast_plotter(star, starage=None, irdap_folder="irdap", method="classical", survey='both'):    
    if survey == "shine":
        year = "20?[!23]"
    elif survey == "snap-shine":
        year = "202[23]"
    else:
        year = "20??"
    stardist, starmag = query_star_data(star)
    if starage is None:
        starage = get_star_age(star)
    print(f"Query for {star} found. Stardist = {stardist}; Starmag = {starmag}, Starage = {starage}")
    pwd = Path(".")
    safestar = star.replace('__', '_').replace('_', '*')
    classical_pca_file = list(pwd.glob(f"{irdap_folder}/{safestar}.{year}-*/reduced_adi/{method}/{safestar}*ADI*sum.fits"))
    for fitsfile in classical_pca_file:
        with fits.open(fitsfile) as f:
            data = f[0].data
        fitsfile = str(fitsfile).split("/")[-1][:-14]
        print(f"Using {fitsfile} for {star}")
        fluxratio = find_flux_ratio(star, irdap_folder)

        radii, det_mass = contrast_rings(data, fluxratio, starage, stardist, starmag)
        plt.plot(PIX_SCALE*stardist*radii, det_mass[0], linestyle="-.", label="$1\sigma$")
        plt.plot(PIX_SCALE*stardist*radii, det_mass[1], linestyle="--", label="$3\sigma$")
        plt.plot(PIX_SCALE*stardist*radii, det_mass[2], linestyle="-", label="$5\sigma$")
        plt.xlim(1,1e4)
        plt.title(f"{star}")
        plt.xlabel("Distance from primary (au)")
        plt.ylabel("Companion mass ($M_{\\mathrm{Jup}}$)")
        plt.ylim(1, 100)
        plt.legend()
        plt.loglog()
        if not survey== "both":
            plt.savefig(f"results/visibility/RMS_sensitive_mass_{fitsfile}_{survey}.png")
        else:
            plt.savefig(f"results/visibility/RMS_sensitive_mass_{fitsfile}.png")
        plt.clf()


def find_flux_ratio(star, irdap_folder="irdap"):
    '''Get the ratio between the coronagraph and flux images.'''
    pwd = Path(".")
    flux_csv = list(pwd.glob(f"{irdap_folder}/{star}*/calibration/flux/{star}*_reference_flux.csv"))[0]
    with open(flux_csv, "r", newline="") as csvfile:
        csventries = csv.reader(csvfile, delimiter=",")
        csventries.__next__()
        ratio = []
        for row in csventries:
            ratio.append(float(row[10]))
    return np.mean(ratio)


def plot_all(targetlist='results/targets.csv', survey="both", skiprows=1):
    '''Plot all targets in the targetlist.'''
    with open(targetlist, "r", newline="") as csvfile:
        csventries = csv.reader(csvfile, delimiter=",")
        for _ in range(skiprows):
            csventries.__next__()
        for row in csventries:
            if row[0][0] == "#":
                row[0] = row[0][1:]
            rms_contrast_plotter(row[0], int(row[2]), survey=survey)
            if not row[1].startswith('%'):
                rms_contrast_plotter(row[1], int(row[2]), survey=survey)


if __name__ == '__main__':
    plot_all(survey="snap-shine")
