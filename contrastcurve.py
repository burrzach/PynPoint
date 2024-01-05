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
    FakePlanetModule, FalsePositiveModule, PcaPsfSubtractionModule, DerotateAndStackModule
from pynpoint.util.image import polar_to_cartesian
from pynpoint.core.processing import ProcessingModule
import configparser
from scipy.optimize import root_scalar
import time
import glob

t0 = time.time()

#settings
scale = 1.73 / 290  #arcsec/pixel
radius = 0.035      #arcsec
angle_step = 60     #deg
sep_step = 0.05     #arcsec
inner_radius = 0.15 #arcsec
outer_radius = 0.8  #arcsec
threshold = 3e-7    #-
tolerance = 1e-6    #-
iterations = 1000   #-

psf_folder = '/home/zburr/PynPoint/6-15-2/'


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
def PlanetInjection(mag, pipeline, science_image, sep, angle, pos_pix, threshold, subtract=False):
    #inject fake planet
    module = FakePlanetModule(name_in='fake',
                              image_in_tag=science_image, 
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
        
        module = ReshapeModule(name_in='reshape_resid', 
                               image_in_tag='residuals', 
                               image_out_tag='resid_reshape', 
                               shape=(39,1,290,290))
        pipeline.add_module(module)
        pipeline.run_module('reshape_resid')
        
        science_image = 'resid_reshape'
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
    
    return fpf - threshold
        

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


## Prepare science image ##
module = DerotateAndStackModule(name_in='derotate_science',
                                image_in_tag='science',
                                image_out_tag='science_derot',
                                derotate=True,
                                stack=None)
pipeline.add_module(module)
pipeline.run_module('derotate_science')

pipeline.set_attribute(data_tag='science_derot', 
                       attr_name='PARANG', 
                       attr_value=np.array([0.]),
                       static=False)


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


## Remove any existing planets ##
science_image = 'science_derot'
image_out = 'removed_1'
count = 1
comp_list = glob.glob(folder+'*companion*_data.txt')
for file in comp_list:
    count += 1
    
    comp_data = np.genfromtxt(file)
    sep = comp_data[0,7]
    angle = comp_data[0,9]
    mag = comp_data[0,13]
    
    module = FakePlanetModule(name_in='remove_companion', 
                              image_in_tag=science_image, 
                              psf_in_tag='planet', 
                              image_out_tag=image_out, 
                              position=(sep,angle), 
                              magnitude=mag,
                              psf_scaling=-1.)
    pipeline.add_module(module)
    pipeline.run_module('remove_companion')
    
    science_image = image_out
    image_out = image_out[:-1] + str(count)


## Setup variables for loop ##
sep_space = np.arange(inner_radius, outer_radius, sep_step)
angle_space = np.arange(0., 360., angle_step)

contrast_map_pre = np.full((len(sep_space), len(angle_space)), np.nan)
contrast_map_post = np.full((len(sep_space), len(angle_space)), np.nan)

initial_guess_pre = 4.
initial_guess_post = 6.

## Loop through each separation and angle ##
for i, sep in enumerate(sep_space):
    for j, angle in enumerate(angle_space):
        print('\n-------------------------------------')
        print('Beginning iterations for:', (sep, angle))
        print('-------------------------------------')
        
        #convert separation and angle to pixel position
        pic = pipeline.get_data(science_image)
        sep_pix = sep / scale
        y,x = polar_to_cartesian(pic, sep_pix, angle)
        pos_pix = (x,y)
        
        #optimize to find brightness at threshold
        #before subtraction
        res = root_scalar(PlanetInjection, 
                          args=(pipeline, science_image, sep, angle, pos_pix, threshold, False),
                          bracket=(0.,15.),
                          x0=initial_guess_pre,
                          rtol=tolerance,
                          maxiter=iterations)
        
        print(res)
        contrast_map_pre[i,j] = res.root #save result
        initial_guess_pre = contrast_map_pre[i,j] #initial guess for next position
        
        #after subtraction
        res = root_scalar(PlanetInjection, 
                          args=(pipeline, science_image, sep, angle, pos_pix, threshold, True),
                          bracket=(initial_guess_pre - 4., 20.),
                          x0=initial_guess_post,
                          rtol=tolerance,
                          maxiter=iterations)
        
        print(res)
        contrast_map_post[i,j] = res.root #save result
        initial_guess_post = contrast_map_post[i,j] #initial guess for next position

#save data
seps = np.full((len(sep_space)+1, 1), np.nan)
for i, el in enumerate(sep_space):
    seps[i+1, 0] = el
    
data1 = np.vstack((angle_space, contrast_map_pre))
data1 = np.hstack((seps, data1))

data2 = np.vstack((angle_space, contrast_map_post))
data2 = np.hstack((seps, data2))

data = np.vstack((data1, data2))
np.savetxt(folder+'contrast_map.txt', data)

t1 = time.time()
dt = (t1 - t0)
print(f'\nContrast curves completed after {int(dt//60)}minutes {int(dt%60)}seconds.')
