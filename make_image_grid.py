#imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import axes_grid
from pynpoint import Pypeline, FitsReadingModule
from pynpoint.util.image import polar_to_cartesian
from math import ceil
import pandas as pd


#initialize
folder = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/cubes/"
props_file = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/star_data.csv"
output_folder = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/cubes/"
companions_folder = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/companions/"

pipeline = Pypeline(working_place_in=folder,
                    input_place_in=folder,
                    output_place_in=folder)

#settings
star_props = pd.read_csv(props_file, index_col=0)
star_props = star_props.drop_duplicates(subset='2MASS', keep='last')
observations = np.array(star_props['obs'])
observations = np.array(["2023-05-27",    #just systems with candidates
                         #"2023-05-30-2", 
                         #"2023-06-15-1",
                         #"2023-07-26-1",
                         #"2023-08-07-2"
                          ])
obs_per_image = 9 #number of images to plot in each grid
ncols = 3         #number of columns worth of images in each grid

companion_list = {"2023-05-27":  1, #how many companions are in each system
                  "2023-05-30-2":1, 
                  "2023-06-15-1":1,
                  "2023-07-26-1":1,
                  "2023-08-07-2":2}
radius = 0.035
bi_radius = 0.05
scale = 1.73 / 290

#loop through each image
image_indices = range(0,len(observations),obs_per_image)
grid = 0
for n in range(len(image_indices)):
    grid += 1
    if n == len(image_indices)-1:
        subset = observations[image_indices[n]:]
    else:
        subset = observations[image_indices[n]:image_indices[n+1]]
    
    #create figure
    nrows = ceil(len(subset) / ncols)
    nsubplts = 2
    fig = plt.figure()
    
    #loop through observations in this subset
    for i, ob in enumerate(subset):
        #make subfigure for this observation
        ag = axes_grid.Grid(fig, (nrows, ncols, i+1), (1, nsubplts), axes_pad=0)
        ag[0].set_title(ob)
        #plt.title(ob)
        
        #load raw
        try:
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
            
            raw = pipeline.get_data('raw')
            resid = pipeline.get_data('resid')
            
        except:
            raw = np.zeros((1,290,290))
            resid = np.zeros((1,290,290))
        
        #make subfigure
        #plot raw
        size = scale * raw.shape[-1]/2.
        
        ag[0].imshow(raw[0], origin='lower', extent=[size, -size, -size, size])
        #cb = ag[0].colorbar()
        #cb.set_label('Flux (ADU)', size=14.)
        
        #plot resid
        size = scale * resid.shape[-1]/2.
        
        ag[1].imshow(resid[0], origin='lower', extent=[size, -size, -size, size])
        #cb = ag[1].colorbar()
        #cb.set_label('Flux (ADU)', size=14.)
        
        if ob in companion_list.keys():
            if ob == '2023-06-15-1':
                app_rad = bi_radius #/ scale
            else:
                app_rad = radius #/ scale
            n_companions = companion_list[ob]
            for j in range(1, n_companions+1):
                comp_file = companions_folder + ob + f"_companion{j}_data.txt"
                data = np.genfromtxt(comp_file)
                
                sep = data[0,7]
                angle = data[0,9]
                pos = polar_to_cartesian(raw[0], sep/scale, angle)
                y = (pos[0] - raw.shape[-1]/2) * scale
                x = (pos[1] - raw.shape[-1]/2) * -scale
                
                circle0 = plt.Circle((x,y), radius=app_rad, color='red',
                                    fill=False, ls=':')
                ag[0].add_patch(circle0)
                
                circle1 = plt.Circle((x,y), radius=app_rad, color='red',
                                    fill=False, ls=':')
                ag[1].add_patch(circle1)
        
    #plt.savefig(output_folder + f'subtraction_grid{grid}', bbox_inches='tight')
