################################
# Performs SDI on science cube #
# using Pynpoint package       #
################################


# #imports
#import os
#import sys

# #check for arguments
# if len(sys.argv) > 1:
#     folder = sys.argv[1]
#     if not os.path.exists(folder):
#         raise OSError(folder + " does not exist.")
# else:
#     raise ValueError("No dir given.")

#import the rest of the modules (pynpoint import is a bit slow, so import after checking for valid path)
#import urllib
#import matplotlib.pyplot as plt
import numpy as np
from pynpoint import Pypeline, FitsReadingModule, ParangReadingModule, WavelengthReadingModule,\
    PSFpreparationModule, PcaPsfSubtractionModule, FitsWritingModule, FakePlanetModule,\
    AddLinesModule, RemoveFramesModule, StarExtractionModule, FalsePositiveModule, TextWritingModule,\
    AttributeWritingModule
from pynpoint.core.processing import ProcessingModule
from pynpoint.util.image import polar_to_cartesian

#folder = "D:\\Zach\\Documents\\TUDelft\\MSc\\Thesis\\PynPoint\\6-15-2\\"
#psffolder = "D:\\Zach\\Documents\\TUDelft\\MSc\\Thesis\\PynPoint\\7-26-1\\"
folder = "/home/zburr/PynPoint/6-15-2/"
psffolder = "/home/zburr/PynPoint/7-26-1/"

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
        
        self.m_out_port.copy_attributes(self.m_in_port)
        self.m_out_port.add_history('ReshapeModule', str(orig_shape)+'->'+str(self.shape))

        self.m_out_port.close_port()


#Initialize pipeline object
pipeline = Pypeline(working_place_in=folder,
                    input_place_in  =folder,
                    output_place_in =folder)


#Read in science data
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


#Read in psf
module = FitsReadingModule(name_in='readpsf',
                            image_tag='psf',
                            filenames=[psffolder+'science_cube.fits'],
                            input_dir=None,
                            ifs_data=True)
pipeline.add_module(module)

module = ParangReadingModule(name_in='parangpsf',
                              data_tag='psf',
                              file_name=psffolder+'science_derot.fits')
pipeline.add_module(module)

module = WavelengthReadingModule(name_in='wavelengthpsf',
                                  data_tag='psf',
                                  file_name=folder+'wavelength.fits')
pipeline.add_module(module)


# #Prepare psf for injecting as fake planet
# #reshape to 3D
# module = ReshapeModule(name_in='shape_down',
#                        image_in_tag='psf',
#                        image_out_tag='psf3D',
#                        shape=(39,80,80))
# pipeline.add_module(module)
# 
# #select just least noisy frame
# module = RemoveFramesModule(name_in='slice_psf', 
#                             image_in_tag='psf3D', 
#                             selected_out_tag='other_psf', 
#                             removed_out_tag='planet', 
#                             frames=[25])
# pipeline.add_module(module)
# 
# #mask out noise
# module = PSFpreparationModule(name_in='maskpsf', 
#                               image_in_tag='planet', 
#                               image_out_tag='masked_planet',
#                               cent_size=None,
#                               edge_size=0.5)
# pipeline.add_module(module)
# 
# #pad to get correct shape
# module = AddLinesModule(name_in='pad', 
#                         image_in_tag='masked_planet', 
#                         image_out_tag='psf_resize', 
#                         lines=(105,105,105,105))
# pipeline.add_module(module)

#Crop planet from image to use as fake planet
module = ReshapeModule(name_in='shape_down',
                       image_in_tag='psf',
                       image_out_tag='psf3D',
                       shape=(39,290,290))
pipeline.add_module(module)

module = StarExtractionModule(name_in='extract_planet', 
                              image_in_tag='psf3D', 
                              image_out_tag='planet',
                              image_size=0.7,
                              fwhm_star=0.2)
pipeline.add_module(module)

module = AddLinesModule(name_in='pad',
                        image_in_tag='planet', 
                        image_out_tag='psf_resize', 
                        lines=(131,132,131,132))
pipeline.add_module(module)

module = RemoveFramesModule(name_in='slice_psf', 
                            image_in_tag='psf_resize', 
                            selected_out_tag='other_psf', 
                            removed_out_tag='planet_slice', 
                            frames=[25])
pipeline.add_module(module)

module = PSFpreparationModule(name_in='maskpsf', 
                              image_in_tag='planet_slice', 
                              image_out_tag='masked_planet',
                              cent_size=None,
                              edge_size=0.5)
pipeline.add_module(module)

#Add in fake planet
module = ReshapeModule(name_in='shape_down_science',
                       image_in_tag='science',
                       image_out_tag='science3D',
                       shape=(39,290,290))
pipeline.add_module(module)

module = FakePlanetModule(name_in='inject', 
                          image_in_tag='science3D', 
                          psf_in_tag='masked_planet', 
                          image_out_tag='fake', 
                          position=(1.5,90), 
                          magnitude=4.)
pipeline.add_module(module)

module = ReshapeModule(name_in='shape_up_science',
                        image_in_tag='fake',
                        image_out_tag='fake_resize',
                        shape=(39,1,290,290))
pipeline.add_module(module)


#Prepare subtraction
module = PSFpreparationModule(name_in='prep',
                              image_in_tag='fake_resize',
                              image_out_tag='prep',
                              mask_out_tag=None,
                              norm=False,
                              resize=None,
                              cent_size=None,
                              edge_size=None)
pipeline.add_module(module)


#Perform subtraction
module = PcaPsfSubtractionModule(pca_numbers=([5, ]),
                                 name_in='pca',
                                 images_in_tag='prep',
                                 reference_in_tag='prep',
                                 res_median_tag='residuals',
                                 subtract_mean=False,
                                 processing_type='SDI'
                                 )
pipeline.add_module(module)


#Write out data
module = FitsWritingModule(name_in='write_res',
                           data_tag='residuals',
                           file_name='6-15-2_fake_residuals.fits',
                           output_dir=folder)
pipeline.add_module(module)

module = FitsWritingModule(name_in='write_fake', 
                           data_tag='fake', 
                           file_name='6-15-2_fake_raw.fits',
                           output_dir=folder)
pipeline.add_module(module)


#run modules
pipeline.run()


#Measure before and after subtraction
raw = pipeline.get_data('prep')
residual = pipeline.get_data('residuals')

#pos_raw = polar_to_cartesian(raw, 1.5, 90)
#pos_resid = polar_to_cartesian(residual, 1.5, 90)

module = FalsePositiveModule(name_in='measure_raw',
                             image_in_tag='fake',
                             snr_out_tag='raw_snr',
                             position=(178,187),
                             aperture=0.2,
                             ignore=False,
                             optimize=True,
                             offset=10)
pipeline.add_module(module)

module = ReshapeModule(name_in='shape_down_resid',
                       image_in_tag='residuals',
                       image_out_tag='residuals3D',
                       shape=(39,290,290))
pipeline.add_module(module)

module = FalsePositiveModule(name_in='measure_resid',
                             image_in_tag='residuals3D',
                             snr_out_tag='resid_snr',
                             position=(90,145),
                             aperture=0.2,
                             ignore=False,
                             optimize=True,
                             offset=10)
pipeline.add_module(module)

module = TextWritingModule(name_in='write_raw_snr',
                           data_tag='raw_snr',
                           file_name='fake_snr.txt')
pipeline.add_module(module)

module = TextWritingModule(name_in='write_resid_snr',
                           data_tag='resid_snr',
                           file_name='resid_snr.txt')
pipeline.add_module(module)

module = AttributeWritingModule(name_in='write_wl',
                           data_tag='science',
                           attribute='WAVELENGTH',
                           file_name='fake_snr.txt')
pipeline.add_module(module)

pipeline.run_module('measure_raw')
pipeline.run_module('shape_down_resid')
pipeline.run_module('measure_resid')
pipeline.run_module('write_raw_snr')
pipeline.run_module('write_resid_snr')
pipeline.run_module('write_wl')
