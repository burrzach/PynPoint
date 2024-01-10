# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 12:47:26 2024

@author: Zach
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pynpoint import Pypeline, FitsReadingModule, DerotateAndStackModule,\
    ParangReadingModule, WavelengthReadingModule



folder = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/cubes/"
ob = "2023-08-07-2"
angle = 232



#Initialize pipeline
pipeline = Pypeline(working_place_in=folder,
                    input_place_in  =folder,
                    output_place_in =folder)


## Load in science image ##
module = FitsReadingModule(name_in='read',
                           image_tag='science',
                           filenames=[folder+'science_cube.fits'],
                           input_dir=None,
                           ifs_data=True)
pipeline.add_module(module)
pipeline.run_module('read')

module = ParangReadingModule(name_in='parang',
                              data_tag='science',
                              file_name=folder+'science_derot.fits')
pipeline.add_module(module)
pipeline.run_module('parang')

module = WavelengthReadingModule(name_in='wavelength',
                                 data_tag='science',
                                 file_name=folder+'wavelength.fits')
pipeline.add_module(module)
pipeline.run_module('wavelength')


#Derotate image
module = DerotateAndStackModule(name_in='rotate', 
                                image_in_tag='science', 
                                image_out_tag='science_derot',
                                extra_rot=angle)
pipeline.add_module(module)
pipeline.run_module('rotate')


#plot to check
raw = pipeline.get_data('science_derot')
plt.imshow(raw[0])
