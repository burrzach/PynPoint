#imports
import numpy as np
import math
from pynpoint import Pypeline, FitsReadingModule, ParangReadingModule, WavelengthReadingModule,\
    AddFramesModule, RemoveFramesModule, FalsePositiveModule, AperturePhotometryModule, \
    DerotateAndStackModule, FitCenterModule, ShiftImagesModule, SubtractImagesModule, \
    FitsWritingModule, RepeatImagesModule
        
from pynpoint.core.processing import ProcessingModule
import configparser


#folder ="D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/2023-06-15-1/"
folder = "/home/zburr/PynPoint/2023-06-15-1/"


#Set configuration file
config = configparser.ConfigParser()
config.add_section('header')
config.add_section('settings')
config['settings']['PIXSCALE'] = str(1.73/290)

with open(folder+'PynPoint_config.ini', 'w') as configfile:
    config.write(configfile)


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


## Load in images ##
#read in science data
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


## Prepare images ##
#coadd science
module = DerotateAndStackModule(name_in='derotate_science',
                                image_in_tag='science',
                                image_out_tag='science_derot',
                                derotate=True,
                                stack=None)
pipeline.add_module(module)
pipeline.run_module('derotate_science')

module = ReshapeModule(name_in='shape_down_science', 
                       image_in_tag='science_derot', 
                       image_out_tag='science3D', 
                       shape=(39,290,290))
pipeline.add_module(module)
pipeline.run_module('shape_down_science')

module = FitCenterModule(name_in='fit',
                         image_in_tag='science3D',
                         fit_out_tag='science_centering',
                         mask_radii=(None,0.5),
                         sign='negative',
                         model='gaussian',
                         filter_size=None)
pipeline.add_module(module)
pipeline.run_module('fit')

module = ShiftImagesModule(name_in='center', 
                           image_in_tag='science3D', 
                           image_out_tag='science_centered', 
                           shift_xy='science_centering')
pipeline.add_module(module)
pipeline.run_module('center')

module = ReshapeModule(name_in='shape_up_science', 
                       image_in_tag='science_centered', 
                       image_out_tag='science4D', 
                       shape=(39,1,290,290))
pipeline.add_module(module)
pipeline.run_module('shape_up_science')

module = RemoveFramesModule(name_in='slice_science', 
                            image_in_tag='science4D', 
                            selected_out_tag='science_sliced', 
                            removed_out_tag='trash', 
                            frames=[0,1,37,38])
pipeline.add_module(module)
pipeline.run_module('slice_science')

module = AddFramesModule(name_in='coadd_science', 
                         image_in_tag='science_sliced', 
                         image_out_tag='science_coadd')
pipeline.add_module(module)
pipeline.run_module('coadd_science')


## Subtract out binary ##
module = RepeatImagesModule(name_in='repeat', 
                            image_in_tag='science_coadd', 
                            image_out_tag='coadd_repeat', 
                            repeat=38)
pipeline.add_module(module)
pipeline.run_module('repeat')

module = SubtractImagesModule(name_in='subtract',
                              image_in_tags=('science_centered', 'coadd_repeat'),
                              image_out_tag='science_subtracted')
pipeline.add_module(module)
pipeline.run_module('subtract')

module = FitsWritingModule(name_in='write',
                           data_tag='science_subtracted',
                           file_name=folder+'binary_subtracted.fits')
pipeline.add_module(module)
pipeline.run_module('write')



