################################
# Performs SDI on science cube #
# using Pynpoint package       #
################################


# #imports
import os
import sys

# #check for arguments
# if len(sys.argv) > 1:
#     folder = sys.argv[1]
#     if not os.path.exists(folder):
#         raise OSError(folder + " does not exist.")
# else:
#     raise ValueError("No dir given.")

#import the rest of the modules (pynpoint import is a bit slow, so import after checking for valid path)
import urllib
import matplotlib.pyplot as plt
import numpy as np
from pynpoint import Pypeline, FitsReadingModule, ParangReadingModule, WavelengthReadingModule, PSFpreparationModule, PcaPsfSubtractionModule
from pynpoint import FitsWritingModule, FakePlanetModule, AddLinesModule
from pynpoint.core.processing import ProcessingModule

folder = "D:\\Zach\\Documents\\TUDelft\\MSc\\Thesis\\YSES_IFU\\2nd_epoch\\2023-07-26-1\\"
#folder = "/data/zburr/yses_ifu/2nd_epoch/processed/2023-07-26-1/products/"

#make module to reshape arrays to drop extra dimension
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


#read in psf
module = FitsReadingModule(name_in='readpsf',
                           image_tag='psf',
                           filenames=[folder+'psf_cube.fits'],
                           input_dir=None,
                           ifs_data=True)
pipeline.add_module(module)

module = ParangReadingModule(name_in='parangpsf',
                             data_tag='psf',
                             file_name=folder+'psf_derot.fits')
pipeline.add_module(module)

module = WavelengthReadingModule(name_in='wavelength',
                                 data_tag='psf',
                                 file_name=folder+'wavelength.fits')
pipeline.add_module(module)


#pad psf to make correct dimensions
module = ReshapeModule(name_in='shape_down',
                       image_in_tag='psf',
                       image_out_tag='psf3D',
                       shape=(39,80,80))
pipeline.add_module(module)

module = AddLinesModule(name_in='pad', 
                        image_in_tag='psf3D', 
                        image_out_tag='psf_resize', 
                        lines=(105,105,105,105))
pipeline.add_module(module)

# module = ReshapeModule(name_in='shape_up',
#                        image_in_tag='psf_resize',
#                        image_out_tag='psf_resize',
#                        shape=(39,1,290,290))
# pipeline.add_module(module)


#add in fake planet
module = ReshapeModule(name_in='shape_down_science',
                       image_in_tag='science',
                       image_out_tag='science3D',
                       shape=(39,290,290))
pipeline.add_module(module)

module = FakePlanetModule(name_in='inject', 
                          image_in_tag='science3D', 
                          psf_in_tag='psf_resize', 
                          image_out_tag='fake', 
                          position=(0.5,270), 
                          magnitude=2.)
pipeline.add_module(module)

module = ReshapeModule(name_in='shape_up_science',
                        image_in_tag='fake',
                        image_out_tag='fake_resize',
                        shape=(39,1,290,290))
pipeline.add_module(module)

#prepare subtraction
module = PSFpreparationModule(name_in='prep',
                              image_in_tag='fake_resize',
                              image_out_tag='prep',
                              mask_out_tag=None,
                              norm=False,
                              resize=None,
                              cent_size=None,
                              edge_size=None)
pipeline.add_module(module)

module = WavelengthReadingModule(name_in='wavelength2',
                                 data_tag='prep',
                                 file_name=folder+'wavelength.fits')
pipeline.add_module(module)

#perform subtraction
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

