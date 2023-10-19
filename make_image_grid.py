#imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import axes_grid
from pynpoint import Pypeline, FitsReadingModule
from math import ceil

#%%
#initialize
folder = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/cubes/"

pipeline = Pypeline(working_place_in=folder,
                    input_place_in=folder,
                    output_place_in=folder)

#load list of observations
observations = np.genfromtxt("D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/observations.csv", 
                             skip_header=1, dtype=str)
obs_per_image = 9
image_indices = range(0,len(observations),obs_per_image)

#loop through each image
for n in range(len(image_indices)):
    if n == len(image_indices)-1:
        subset = observations[image_indices[n]:]
    else:
        subset = observations[image_indices[n]:image_indices[n+1]]
    
    #create figure
    ncols = 3
    nrows = ceil(len(subset) / ncols)
    nsubplts = 2
    fig = plt.figure()
    
    #loop through observations in this subset
    for i, ob in enumerate(subset):
        #make subfigure for this observation
        ag = axes_grid.Grid(fig, (nrows, ncols, i+1), (1, nsubplts), axes_pad=0)
        ag[0].set_title(ob)
        
        #load raw
        module = FitsReadingModule(name_in='read_raw',
                                   input_dir=None,
                                   filenames=[folder+'raw_coadd/'+ob+"_raw_coadd.fits"],
                                   image_tag='raw')
        pipeline.add_module(module)
        pipeline.run_module('read_raw')
        
        #load resid
        module = FitsReadingModule(name_in='read_resid',
                                   input_dir=None,
                                   filenames=[folder+'resid_coadd/'+ob+"_resid_coadd.fits"],
                                   image_tag='resid')
        pipeline.add_module(module)
        pipeline.run_module('read_resid')
        
        #make subfigure
        #plot raw
        raw = pipeline.get_data('raw')
        pixscale = pipeline.get_attribute('raw', 'PIXSCALE')
        size = pixscale * raw.shape[-1]/2.
        
        ag[0].imshow(raw[0], origin='lower', extent=[size, -size, -size, size])
        #cb = ag[0].colorbar()
        #cb.set_label('Flux (ADU)', size=14.)
        
        #plot resid
        resid = pipeline.get_data('resid')
        pixscale = pipeline.get_attribute('resid', 'PIXSCALE')
        size = pixscale * resid.shape[-1]/2.
        
        ag[1].imshow(resid[0], origin='lower', extent=[size, -size, -size, size])
        #cb = ag[1].colorbar()
        #cb.set_label('Flux (ADU)', size=14.)


#%%
obs = "2023-05-27"
file = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/companions/"+obs+"_companion_data.txt"
data = np.genfromtxt(file)

head = data[:5]
print('sep, angle, mag, snr, fpf')
if len(data[0]) > 3:
    for i in range(len(data[0])-2):
        print('companion '+str(i+1)+':\n', head[:,i+2])
else:
    print(head[:,2])
    
spectra = data[5:]
#spectra = spectra[spectra[:,0].argsort()]
wl = spectra[:,0] / 1e3

plt.figure()
plt.plot(wl, spectra[:,1], marker='o', label='host star')
if len(data[0]) > 3:
    for i in range(len(data[0])-2):
        plt.plot(wl, spectra[:,i+2], marker='o', label='companion '+str(i+1))
else:
    plt.plot(wl, spectra[:,2], marker='o', label='companion')

plt.legend()
plt.yscale('log')
plt.xlabel('$\lambda$ $[\mu m]$')
plt.title(obs)
plt.show()
