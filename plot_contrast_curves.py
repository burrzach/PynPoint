#imports
import csv
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D

#settings
plot_ind = False
plot_planets = True

props_file = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/star_data.csv"
curves_folder = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/contrast_curves/"
tracks_folder = "D:/Zach/Documents/TUDelft/MSc/Thesis/phillips2020/" + \
    "evolutionary_tracks/ATMO_2020/ATMO_CEQ/SPHERE_IRDIS/"
companions_folder = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/companions/"
planets_file = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/all_exoplanets_data.csv"

psf_obs = '2023-06-15-2'
M_SUN_TO_M_JUP = 1047.34

companion_list = {"2023-05-27":  2, #how many companions are in each system
                  "2023-05-30-2":1, 
                  "2023-06-15-1":1,
                  "2023-07-26-1":1,
                  "2023-08-07-2":2}

obs_per_image = 9 #number of images to plot in each grid
ncols = 3         #number of columns worth of images in each grid
mass_ticks = [100, 50, 25, 12, 6] #what values of mass to put tick marks for

#load properties and drop duplicates
star_props = pd.read_csv(props_file, index_col=0)
star_props = star_props.drop_duplicates(subset='2MASS', keep='last')
observations = np.array(star_props['obs'])
#observations = np.array(['2023-07-08-2', '2023-07-22-1', '2023-07-22-3'])

plt.rcParams.update({'font.size': 15})

planet_markers = {'Imaging':'o',
                  'Microlensing':'s',
                  'Radial Velocity':'^',
                  'Transit':'p',
                  'Solar System Planet':'o'}
planet_colors = {'Imaging':'blue',
                 'Microlensing':'grey',
                 'Radial Velocity':'orange',
                 'Transit':'purple',
                 'Solar System Planet':'yellow'}

#functions
def apparent_to_absolute(mag, distance):
    return mag - 5 * np.log10(distance) + 5

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

def mag_mass_relate(age_system, track_dir="doc/SPHERE_IRDIS_vega/"):
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
    mag2mass: func
        Scipy interpolator which will convert magnitude to mass
    mass2mag: func
        Scipy interpolator which will convert mass to magnitude
    '''
    mag_to_jupmass = []
    _, evolutionary_tracks = open_evolutionary_tracks(track_dir)

    # Magic numbers used: ind 0=mass (M_sun), 1=age (Gyr), 18=J band magnitude

    for model in evolutionary_tracks:
        # Model age is in Gyr, age_system is in Myr
        ind = np.searchsorted(1000*model[:,1], age_system)
        if ind == len(model):
            ind -= 1
        elif (ind+1 < len(model)) and ((model[ind+1,1] - age_system) < (age_system - model[ind,1])):
            ind+=1  # Check if the value to the right is closer
        if not model[ind,18] == 0:
            mag_to_jupmass.append([model[ind,18], model[ind,0]*M_SUN_TO_M_JUP])

    mag_to_jupmass = np.array(mag_to_jupmass)
    mag_to_jupmass = mag_to_jupmass[np.argsort(mag_to_jupmass[:,1])]
    mag_to_jupmass = mag_to_jupmass[::-1] # np.interp needs a monotonically increasing array
    mag2mass = interp1d(mag_to_jupmass[:,0], mag_to_jupmass[:,1], kind='linear', 
                        fill_value='extrapolate')
    mass2mag = interp1d(mag_to_jupmass[:,1], mag_to_jupmass[:,0], kind='linear', 
                        fill_value='extrapolate')
    return mag2mass, mass2mag

#%%
#load star properties
fake_app_mag = star_props.loc[star_props['obs'] == psf_obs, 'J'].iloc[0]
fake_err = star_props.loc[star_props['obs'] == psf_obs, 'J_err'].iloc[0]

#make big image
fig, axes = plt.subplots(2, 1, sharex=True, num='all_curves', clear=True)
fig.set_figheight(25)
fig.set_figwidth(11)

axes[0].invert_yaxis()
axes[0].set_xlim(xmin=0, xmax=0.8)
axes[0].set_ylabel('Absolute magnitude [-]')

#axes[1].set_ylim(ymax=100)
axes[1].set_yscale('log')
axes[1].set_xlabel('Separation [arcsec]')
axes[1].set_ylabel('Mass [M$_{Jup}$]')

#make big image with planets
#0 id, 1 planet name, 2 hostname, 3 method, 4 a, 5 mass
planet_data = np.genfromtxt(planets_file, delimiter=',', missing_values='',
                            filling_values=np.nan, dtype=str)
plt.figure(num='all_planets', clear=True)
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Mass [M$_{Jup}$]')
plt.xlabel('Semi-Major Axis [AU]')

legend_elements = [Line2D([0],[0], color='w', label='Method of Discovery')]
for method in planet_markers.keys():
    legend_elements.append(Line2D([0], [0], color='w', 
                                  markerfacecolor=planet_colors[method],
                                  marker=planet_markers[method], markersize=15,
                                  label=method, markeredgecolor='black'))
plt.legend(handles=legend_elements, loc='lower right')
    
for planet in planet_data[:-8]:
    method = planet[3]
    pl_a = float(planet[4])
    pl_mass = float(planet[5])
    if method in planet_markers.keys():
        plt.scatter(pl_a, pl_mass, marker=planet_markers[method], 
                    color=planet_colors[method], edgecolor='black', s=75)
for planet in planet_data[-8:]:
    pl_a = float(planet[4])
    pl_mass = float(planet[5])
    plt.scatter(pl_a, pl_mass, marker=planet_markers['Solar System Planet'], 
                color=planet_colors['Solar System Planet'], s=100, edgecolor='black')
    if planet[1] == 'Venus' or planet[1] == 'Uranus':
        plt.annotate(planet[1], (pl_a, pl_mass), textcoords='offset fontsize',
                     xytext=(0,-1.1))
    else:
        plt.annotate(planet[1], (pl_a, pl_mass))
            
max_mass = plt.gca().get_ylim()[1]
alpha = 0.05 #1 / (len(observations) - len(companion_list))

plt.annotate('Region the YSES IFS observations\nare sensitive to', (40,17),
              xytext=(300,0.5), arrowprops=dict(lw=2, color='black'))

#%%
#loop through each image
for obs in observations:
    #load data
    file = curves_folder + '/data/' + obs + '_contrast_map.txt'
    try:
        contrast_map = np.genfromtxt(file)
    except:
        continue
    
    seps = int((contrast_map.shape[0] / 2))
    sep_space = contrast_map[1:seps, 0]
    angle_space = contrast_map[0, 1:]
    
    pre_map = contrast_map[1:seps, 1:]
    post_map = contrast_map[seps+1:, 1:]
    
    #grab properties
    dist = star_props.loc[star_props['obs'] == obs, 'dist'].iloc[0]
    age = star_props.loc[star_props['obs'] == obs, 'age'].iloc[0]
    
    mag2mass, mass2mag = mag_mass_relate(age, tracks_folder)
    
    #calculations
    pre_curve = np.nanmean(pre_map, 1) + fake_app_mag
    pre_error = np.nanstd(pre_map, 1) + fake_err
    abs_pre_curve = apparent_to_absolute(pre_curve, dist)
    
    #mass_pre_curve = mag2mass(abs_pre_curve)
    
    post_curve = np.nanmean(post_map, 1) + fake_app_mag
    post_error = np.nanstd(post_map, 1) + fake_err
    abs_post_curve = apparent_to_absolute(post_curve, dist)
    
    mass_post_curve = mag2mass(abs_post_curve)
    sep_au = sep_space * dist
    
    #plot in own image
    if plot_ind:
        fig, ax1 = plt.subplots(1, 1, num=obs+'_contrast_curve', clear=True)    
        ax1.errorbar(sep_space, abs_pre_curve, yerr=pre_error, marker='o', 
                     capsize=3, label='Before SDI')
        ax1.errorbar(sep_space, abs_post_curve, yerr=post_error, marker='o', 
                     capsize=3, label='After SDI') 
        ax1.invert_yaxis()
        ax1.set_xlim(xmin=0)
        ax1.set_xlabel('Separation [arcsec]')
        ax1.set_ylabel('Absolute magnitude [-]')
        ax1.set_title(obs)
        ax1.legend()
        ax1.yaxis.set_tick_params(labelleft=True)
        
        ax2 = ax1.secondary_yaxis('right', functions=(mag2mass, mass2mag))
        ax2.set_ylabel('Mass [$M_{Jup}$]')
        ax2.set_yticks(mass_ticks)
        
        plt.savefig(curves_folder + '/figures/' + obs + '_contrast_map.png', bbox_inches='tight')
        plt.close()
    
    #check for companions
    if obs in companion_list.keys():
        if plot_ind:
            n_companions = companion_list[obs]
            for j in range(1, n_companions+1):
                comp_file = companions_folder + obs + f"_companion{j}_data.txt"
                data = np.genfromtxt(comp_file)
                
                sep = data[0,7]
                #sep_err = data[0,8]
                
                dmag = data[0,13]
                #dmag_err = ...
                
                star_mag = star_props.loc[star_props['obs'] == psf_obs, 'J'].iloc[0]
                #star_err = star_props.loc[star_props['obs'] == psf_obs, 'J_err'].iloc[0]
                
                mag = apparent_to_absolute(dmag + star_mag, dist)
                #mag_err = apparent_to_absolute(dmag_err + star_err, dist)
                
                if n_companions == 1:
                    label = "Candidate companion"
                else:
                    label = f"Candidate companion {j}"
                # ax1.errorbar(sep, mag, xerr=sep_err, yerr=mag_err, marker='*',
                #              capsize=3, label=label)
                ax1.scatter(sep, mag, marker='*', label=label)
    else:
        #plot in big image
        # plt.errorbar(sep_space, abs_pre_curve, yerr=pre_error, marker='o', 
        #          capsize=0, color='blue', alpha=0.8, markersize=10)
        plt.figure(num='all_curves')
        axes[0].plot(sep_space, abs_post_curve, marker='o', 
                     alpha=0.8, markersize=10, color='orange')
        axes[1].plot(sep_space, mass_post_curve, marker='o',
                     alpha=0.8, markersize=10, color='orange')
        
        if plot_planets:
            plt.figure(num='all_planets')
            plt.fill_between(sep_au, mass_post_curve, max_mass, facecolor='red',
                             alpha=alpha, where=mass_post_curve<max_mass)

#plt.gca().set_ylim(ymin=0.1, ymax=max_mass)
plt.gca().set_ylim(ymax=max_mass)
#plt.gca().set_xlim(xmin=1)
