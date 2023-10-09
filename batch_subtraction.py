################################
# Performs SDI on science cube #
# using Pynpoint package       #
################################


#imports
import os
import sys

#check for arguments
if len(sys.argv) > 1:
    folder = sys.argv[1]
    if not os.path.exists(folder):
        raise OSError(folder + " does not exist.")
else:
    raise ValueError("No dir given.")

#import the rest of the modules (pynpoint import is a bit slow, so import after checking for valid path)
import urllib
import matplotlib.pyplot as plt
from pynpoint import Pypeline, FitsReadingModule, ParangReadingModule, WavelengthReadingModule, PSFpreparationModule, PcaPsfSubtractionModule
from pynpoint import FitsWritingModule


#initialize pipeline object
pipeline = Pypeline(working_place_in=folder,
                    input_place_in  =folder,
                    output_place_in =folder)

#read in science data
module = FitsReadingModule(name_in='read',
                           image_tag='science',
                           filenames=[folder+'science_cube.fits'],
                           input_dir=None,
                           ifs_data=True)
pipeline.add_module(module)

module = ParangReadingModule(name_in='parang',
                             data_tag='science',
                             file_name=folder+'science_derot.fits')
pipeline.add_module(module)

module = WavelengthReadingModule(name_in='wavelength',
                                 data_tag='science',
                                 file_name=folder+'wavelength.fits')
pipeline.add_module(module)


#mask psf
module = PSFpreparationModule(name_in='prep',
                              image_in_tag='science',
                              image_out_tag='prep',
                              mask_out_tag=None,
                              norm=False,
                              resize=None,
                              cent_size=None,
                              edge_size=None)
pipeline.add_module(module)

#prepare subtraction
module = PcaPsfSubtractionModule(pca_numbers=([10, ]),
                                 name_in='pca',
                                 images_in_tag='prep',
                                 reference_in_tag='prep',
                                 res_median_tag='residuals',
                                 subtract_mean=False,
                                 processing_type='SDI'
                                 )
pipeline.add_module(module)


#write out data
module = FitsWritingModule(name_in='write_res',
                           data_tag='residuals',
                           file_name='residuals.fits',
                           output_dir=folder)
pipeline.add_module(module)


#run module
pipeline.run()

