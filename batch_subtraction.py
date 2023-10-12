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
import numpy as np
from pynpoint import Pypeline, FitsReadingModule, ParangReadingModule, \
    WavelengthReadingModule, PSFpreparationModule, PcaPsfSubtractionModule, \
    FitsWritingModule, AddFramesModule, RemoveFramesModule, DerotateAndStackModule
from pynpoint.core.processing import ProcessingModule

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

###############################################################################
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
module = PcaPsfSubtractionModule(pca_numbers=([1, ]),
                                 name_in='pca',
                                 images_in_tag='prep',
                                 reference_in_tag='prep',
                                 res_median_tag='residuals',
                                 subtract_mean=False,
                                 processing_type='SDI'
                                 )
pipeline.add_module(module)


#coadd raw
module = DerotateAndStackModule(name_in='derotate',
                                image_in_tag='science',
                                image_out_tag='science_derot',
                                derotate=True,
                                stack=None)
pipeline.add_module(module)

module = ReshapeModule(name_in='shape_down_science',
                       image_in_tag='science_derot',
                       image_out_tag='science3D',
                       shape=(39,290,290))
pipeline.add_module(module)

module = RemoveFramesModule(name_in='slice_science', 
                            image_in_tag='science3D', 
                            selected_out_tag='science_sliced', 
                            removed_out_tag='trash', 
                            frames=[0,1,37,38])
pipeline.add_module(module)

module = AddFramesModule(name_in='coadd_science', 
                         image_in_tag='science_sliced', 
                         image_out_tag='coadd_raw')
pipeline.add_module(module)


#coadd residuals
module = ReshapeModule(name_in='shape_down_resid',
                       image_in_tag='residuals',
                       image_out_tag='resid3D',
                       shape=(39,290,290))
pipeline.add_module(module)

module = RemoveFramesModule(name_in='slice_resid', 
                            image_in_tag='resid3D', 
                            selected_out_tag='resid_sliced', 
                            removed_out_tag='trash', 
                            frames=[0,1,37,38])
pipeline.add_module(module)

module = AddFramesModule(name_in='coadd_resid', 
                         image_in_tag='resid_sliced', 
                         image_out_tag='coadd_resid')
pipeline.add_module(module)


#write out data
module = FitsWritingModule(name_in='write_res',
                           data_tag='residuals',
                           file_name='residuals.fits',
                           output_dir=folder)
pipeline.add_module(module)

module = FitsWritingModule(name_in='write_coadd',
                           data_tag='coadd_resid',
                           file_name='resid_coadd.fits',
                           output_dir=folder)
pipeline.add_module(module)

module = FitsWritingModule(name_in='write_coadd_raw',
                           data_tag='coadd_raw',
                           file_name='raw_coadd.fits',
                           output_dir=folder)
pipeline.add_module(module)

#run module
pipeline.run()

