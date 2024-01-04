# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 16:42:50 2023

@author: Zach
"""

import numpy as np
from pynpoint import Pypeline, FitsReadingModule, ParangReadingModule, WavelengthReadingModule, \
    PSFpreparationModule, AddLinesModule, RemoveFramesModule, AddFramesModule, \
    FakePlanetModule, FalsePositiveModule, PcaPsfSubtractionModule, DerotateAndStackModule
from pynpoint.util.image import polar_to_cartesian
from pynpoint.core.processing import ProcessingModule
import configparser

scale = 1.73 / 290  #arcsec/pixel
radius = 0.035      #arcsec
angles = [0., 120., 240.]
seps = [0.2, 0.4, 0.6]

psf_folder = '/home/zburr/PynPoint/6-15-2/'
folder = ...
curve = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/contrast_curves/2023-06-15-2_contrast_map.txt"


#Module to reshape arrays (to drop/add extra dimension)
class ReshapeModule(ProcessingModule):

    def __init__(self,
                 name_in,
                 image_in_tag,
                 image_out_tag,
                 shape
                 ):

        super(ReshapeModule, self).__init__(name_in)

        self.m_in_port = self.add_input_port(image_in_tag)
        self.m_out_port = self.add_output_port(image_out_tag)
        
        self.shape = shape

    def run(self):
        
        image = self.m_in_port.get_all()
        orig_shape = np.shape(image)
        
        reshaped = np.reshape(image, self.shape)
        
        self.m_out_port.set_all(reshaped)
        
        history = str(orig_shape)+'->'+str(self.shape)
        self.m_out_port.copy_attributes(self.m_in_port)
        self.m_out_port.add_history('ReshapeModule', history)

        self.m_out_port.close_port()



#Initialize pipeline
pipeline = Pypeline(working_place_in=folder,
                    input_place_in  =folder,
                    output_place_in =folder)

## Load PSF ##
module = FitsReadingModule(name_in='readpsf',
                            image_tag='psf',
                            filenames=[psf_folder+'psf_cube.fits'],
                            input_dir=None,
                            ifs_data=True)
pipeline.add_module(module)
pipeline.run_module('readpsf')

module = ParangReadingModule(name_in='parangpsf',
                              data_tag='psf',
                              file_name=psf_folder+'psf_derot.fits')
pipeline.add_module(module)
pipeline.run_module('parangpsf')

module = WavelengthReadingModule(name_in='wavelengthpsf',
                                  data_tag='psf',
                                  file_name=psf_folder+'wavelength.fits')
pipeline.add_module(module)
pipeline.run_module('wavelengthpsf')


## Prepare PSF for injection ##
module = RemoveFramesModule(name_in='slice_psf', 
                            image_in_tag='psf', 
                            selected_out_tag='other_psf', 
                            removed_out_tag='psf_slice', 
                            frames=[20])
pipeline.add_module(module)
pipeline.run_module('slice_psf')

module = PSFpreparationModule(name_in='maskpsf', 
                              image_in_tag='psf_slice', 
                              image_out_tag='psf_masked',
                              cent_size=None,
                              edge_size=radius)
pipeline.add_module(module)
pipeline.run_module('maskpsf')

module = AddLinesModule(name_in='pad', 
                        image_in_tag='psf_masked', 
                        image_out_tag='psf_pad', 
                        lines=(105,105,105,105))
pipeline.add_module(module)
pipeline.run_module('pad')

module = ReshapeModule(name_in='reshape_psf', 
                       image_in_tag='psf_pad', 
                       image_out_tag='planet', 
                       shape=(1,1,290,290))
pipeline.add_module(module)
pipeline.run_module('reshape_psf')

#Read in data
contrast_map = np.genfromtxt(curve)

seps = int((contrast_map.shape[0] / 2))
sep_space = contrast_map[1:seps, 0]
angle_space = contrast_map[0, 1:]

pre_map = contrast_map[1:seps, 1:]
post_map = contrast_map[seps+1:, 1:]

for ang in angles:
    for sep in seps:
        i_ang = np.argmax(angle_space == ang)
        i_sep = np.argmax(sep_space == sep)
        mag = pre_map[i_sep, i_ang]
        
        #!!! TODO:
        #Next step: inject fake planet with fake planet module
        #Perform subtraction?? Will be difficult if injecting planets radially like this, could do planets in spiral or smt
        #Coadd
        #Output fits file
        #Plot image in matplotlib and save
        #(May need to do injection on server and plotting on laptop.)