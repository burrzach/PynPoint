#To generate contrast curve:
#grab PSF to use as planet
#for each position:
#    for each angle:
#        inject fake planet
#        (perform subtraction)
#        coadd
#        measure snr
#        if close enough to threshold:
#            end loop; output brightness
#        else:
#            update brightness and repeat
#    average brightness at all angles
#output brightness of planet at threshold at each distance

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

#imports
import numpy as np
from pynpoint import Pypeline, FitsReadingModule, ParangReadingModule, WavelengthReadingModule, \
    PSFpreparationModule, AddLinesModule, RemoveFramesModule, AddFramesModule, \
    FakePlanetModule, FalsePositiveModule, PcaPsfSubtractionModule
from pynpoint.util.image import polar_to_cartesian
from pynpoint.core.processing import ProcessingModule
import configparser
from scipy.optimize import minimize


#settings
scale = 1.73 / 290  #arcsec/pixel
radius = 0.035      #arcsec
angle_step = 60     #deg
sep_step = 0.01     #arcsec
inner_radius = 0.05 #arcsec
outer_radius = 0.85 #arcsec
threshold = 3e-7    #-
tolerance = 1e-13   #-
iterations = 1000   #-


#Set configuration file
config = configparser.ConfigParser()
config.add_section('header')
config.add_section('settings')
config['settings']['PIXSCALE'] = str(scale)

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
        

#Function to inject planet of given brightness and find fpf
def PlanetInjection(mag, pipeline, pos_pix, threshold, subtract=False):
    #inject fake planet
    module = FakePlanetModule(name_in='fake',
                              image_in_tag='science', 
                              psf_in_tag='planet', 
                              image_out_tag='injected', 
                              position=(sep,angle), 
                              magnitude=mag)
    pipeline.add_module(module)
    pipeline.run_module('fake')
    
    #perform SDI if needed
    if subtract == True:
        module = PcaPsfSubtractionModule(pca_numbers=([1, ]),
                                         name_in='pca',
                                         images_in_tag='injected',
                                         reference_in_tag='injected',
                                         res_median_tag='residuals',
                                         subtract_mean=False,
                                         processing_type='SDI')
        pipeline.add_module(module)
        pipeline.run_module('pca')
        
        science_image = 'residuals'
    else:
        science_image = 'injected'
    
    #coadd image
    module = RemoveFramesModule(name_in='slice_science', 
                                image_in_tag=science_image, 
                                selected_out_tag='sliced', 
                                removed_out_tag='trash', 
                                frames=[0,1,37,38])
    pipeline.add_module(module)
    pipeline.run_module('slice_science')

    module = AddFramesModule(name_in='coadd_science', 
                             image_in_tag='sliced', 
                             image_out_tag='coadd')
    pipeline.add_module(module)
    pipeline.run_module('coadd_science')
    
    #measure fpf
    module = FalsePositiveModule(name_in='measure_fpf', 
                                 image_in_tag='coadd', 
                                 snr_out_tag='fpf', 
                                 position=pos_pix, 
                                 aperture=radius,
                                 output_noise=False)
    pipeline.add_module(module)
    pipeline.run_module('measure_fpf')
    
    fpf = pipeline.get_data('fpf')[0,5]
    
    return abs(fpf - threshold)
        

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

#read in psf
module = FitsReadingModule(name_in='readpsf',
                            image_tag='psf',
                            filenames=[folder+'psf_cube.fits'],
                            input_dir=None,
                            ifs_data=True)
pipeline.add_module(module)
pipeline.run_module('readpsf')

module = ParangReadingModule(name_in='parangpsf',
                              data_tag='psf',
                              file_name=folder+'psf_derot.fits')
pipeline.add_module(module)
pipeline.run_module('parangpsf')

module = WavelengthReadingModule(name_in='wavelengthpsf',
                                  data_tag='psf',
                                  file_name=folder+'wavelength.fits')
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
                        image_out_tag='planet', 
                        lines=(105,105,105,105))
pipeline.add_module(module)
pipeline.run_module('pad')


## Loop through each separation and angle ##
sep_space = np.arange(inner_radius, outer_radius, sep_step)
angle_space = np.arange(0., 360., angle_step)
contrast_map = np.zeros((len(sep_space), len(angle_space)))
for i, sep in enumerate(sep_space):
    for j, angle in enumerate(angle_space):
        print('Beginning iterations for:', (sep, angle))
        
        #convert polar position into pixels
        pic = pipeline.get_data('science')
        sep_pix = sep / scale
        pos_pix = polar_to_cartesian(pic, sep_pix, angle)
        
        #optimize to find brightness at threshold
        res = minimize(PlanetInjection, 
                       x0=4., 
                       args=(pipeline, pos_pix, threshold, False),
                       method='SLSQP',
                       bounds=[(0.,15)],
                       tol=tolerance,
                       options={'maxiter':iterations, 'disp':True})
        
        #grab results
        iterations = res.nit
        success = res.success
        message = res.message
        brightness = res.x
        
        if success:
            print(f'After {iterations}, optimization terminated successfully.\
                  magnitude at ({sep},{angle}) is {brightness}')
        print(message)
        
        contrast_map[i,j] = brightness
        