#imports
import numpy as np
import math
from pynpoint import Pypeline, FitsReadingModule, ParangReadingModule, WavelengthReadingModule,\
    AddFramesModule, RemoveFramesModule, FalsePositiveModule, AperturePhotometryModule, \
    DerotateAndStackModule, FitCenterModule, FakePlanetModule, PSFpreparationModule, \
    AddLinesModule
from pynpoint.core.processing import ProcessingModule
from pynpoint.util.image import cartesian_to_polar, center_subpixel
import configparser


#Settings
obs = '2023-05-27'
pos_guess = [(247., 146.), (253., 162.)] #2023-05-27
#pos_guess = [(211.5, 176.5)] #2023-05-30-2
#pos_guess = [(125., 170.)] #2023-06-15-1 #binary companion
#pos_guess = [(109., 58.)] #2023-07-26-1
#pos_guess = [(213.,90.), (224.,85.)] #2023-08-07-2
radius = 0.035
#radius = 0.05 #larger radius for binary companion
scale = 1.73 / 290

folder = "/data/zburr/yses_ifu/2nd_epoch/processed/"+obs+"/products/"


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


#Initialize pipeline
pipeline = Pypeline(working_place_in=folder,
                    input_place_in  =folder,
                    output_place_in =folder)


## Load in images ##
#read in science data
module = FitsReadingModule(name_in='read',
                           image_tag='science_reshape',
                           filenames=[folder+'science_cube.fits'],
                           input_dir=None,
                           ifs_data=True)
pipeline.add_module(module)
pipeline.run_module('read')

# module = ReshapeModule(name_in='reshape_science',  #for residuals, because they are 1x39x290x290
#                        image_in_tag='science', 
#                        image_out_tag='science_reshape', 
#                        shape=(39,1,290,290))
# pipeline.add_module(module)
# pipeline.run_module('reshape_science')

module = ParangReadingModule(name_in='parang',
                              data_tag='science_reshape',
                              file_name=folder+'science_derot.fits')
pipeline.add_module(module)
pipeline.run_module('parang')
# pipeline.set_attribute('science_reshape', 'PARANG', [0.], static=False) #for when using (derotated) resids

module = WavelengthReadingModule(name_in='wavelength',
                                 data_tag='science_reshape',
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


## Prepare images ##
#coadd science
module = DerotateAndStackModule(name_in='derotate_science',
                                image_in_tag='science_reshape',
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

module = RemoveFramesModule(name_in='slice_science', 
                            image_in_tag='science_derot', 
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

pipeline.set_attribute(data_tag='science_coadd', 
                       attr_name='PARANG', 
                       attr_value=np.array([0.]),
                       static=False)

#coadd psf
module = DerotateAndStackModule(name_in='derotate_psf',
                                image_in_tag='psf',
                                image_out_tag='psf_derot',
                                derotate=True,
                                stack='median',
                                dimension='time')
pipeline.add_module(module)
pipeline.run_module('derotate_psf')

module = RemoveFramesModule(name_in='slice_psf', 
                            image_in_tag='psf_derot', 
                            selected_out_tag='psf_sliced', 
                            removed_out_tag='trash', 
                            frames=[0,1,37,38])
pipeline.add_module(module)
pipeline.run_module('slice_psf')

module = AddFramesModule(name_in='coadd_psf', 
                         image_in_tag='psf_sliced', 
                         image_out_tag='psf_coadd')
pipeline.add_module(module)
pipeline.run_module('coadd_psf')

module = ReshapeModule(name_in='shape_down_psf', 
                       image_in_tag='psf_derot', 
                       image_out_tag='psf3D', 
                       shape=(39,80,80))
pipeline.add_module(module)
pipeline.run_module('shape_down_psf')

#prepare psf for injecting as fake planet
module = PSFpreparationModule(name_in='maskpsf', 
                              image_in_tag='psf_coadd', 
                              image_out_tag='psf_masked',
                              cent_size=None,
                              edge_size=0.045)
pipeline.add_module(module)
pipeline.run_module('maskpsf')

module = AddLinesModule(name_in='pad', 
                        image_in_tag='psf_masked', 
                        image_out_tag='planet', 
                        lines=(105,105,105,105))
pipeline.add_module(module)
pipeline.run_module('pad')


## Measure star spectrum ##
wl = pipeline.get_attribute('psf', 'WAVELENGTH', static=False)

module = AperturePhotometryModule(name_in='measure_star', 
                                  image_in_tag='psf3D', 
                                  phot_out_tag='star_phot',
                                  radius=0.035,
                                  position=None)
pipeline.add_module(module)
pipeline.run_module('measure_star')

star_spectrum = pipeline.get_data('star_phot')[:,0]
star_tot = sum(star_spectrum[2:-2])


pic = pipeline.get_data('science_coadd')
center = center_subpixel(pic)


## Loop for multiple companions ##
for i, guess in enumerate(pos_guess):
    #initialize array
    data = np.full((40,14), np.nan)
    data[0,1] = star_tot
    data[1:,0] = wl
    data[1:,1] = star_spectrum
    
    #remove all other planets
    science_image = 'science_coadd'
    image_out = 'removed1'
    count = 1
    for j, guess2 in enumerate(pos_guess):
        if j != i:
            count += 1
            inject_pos = cartesian_to_polar(center, guess2[1], guess2[0])
            inject_pos = (inject_pos[0]*scale, inject_pos[1])
            
            #measure companion to be removed
            module = AperturePhotometryModule(name_in='measure_companion', 
                                              image_in_tag='science3D', 
                                              phot_out_tag='companion_phot',
                                              radius=radius,
                                              position=guess2)
            pipeline.add_module(module)
            pipeline.run_module('measure_companion')
            
            phot = pipeline.get_data('companion_phot')[:,0]
            
            #calculate mag
            companion_tot = sum(phot[2:-2])
            try:
                mag = -2.5*math.log10(companion_tot/star_tot)
            except:
                mag = 1.
            
            #inject negative fake planet of same magnitude
            module = FakePlanetModule(name_in='fake',
                                      image_in_tag=science_image, 
                                      psf_in_tag='planet', 
                                      image_out_tag=image_out, 
                                      position=inject_pos, 
                                      magnitude=mag,
                                      psf_scaling=-1.)
            pipeline.add_module(module)
            pipeline.run_module('fake')
            
            science_image = image_out
            image_out = image_out[:-1] + str(count)
            
            
    #find planet position
    angle = cartesian_to_polar(center, guess[1], guess[0])[1]
    module = FitCenterModule(name_in='fit',
                             image_in_tag=science_image,
                             fit_out_tag='companion_pos',
                             mask_radii=(None,0.5),
                             sign='positive',
                             model='gaussian',
                             filter_size=0.01,
                             guess=(guess[0]-center[0], guess[1]-center[1], 5., 5., 5000., 0., 0.))
    pipeline.add_module(module)
    pipeline.run_module('fit')

    comp_fit = pipeline.get_data('companion_pos')[0]
    pos_pix = (comp_fit[0]+center[0], comp_fit[2]+center[1])
    pos_pol = np.array(cartesian_to_polar(center, pos_pix[1], pos_pix[0]))
    pos_pol[0] *= scale

    err1 = np.array(cartesian_to_polar(center, pos_pix[1]-comp_fit[3], pos_pix[0]-comp_fit[1]))
    err2 = np.array(cartesian_to_polar(center, pos_pix[1]+comp_fit[3], pos_pix[0]+comp_fit[1]))
    pos_err = abs(err1 - err2)
    pos_err[0] *= scale
    
    data[0,  7] = pos_pol[0] #sep
    data[0,  8] = pos_err[0] #sep error
    data[0,  9] = pos_pol[1] #angle
    data[0, 10] = pos_err[1] #angle error
    
    
    #measure companion stats in coadd
    module = FalsePositiveModule(name_in='snr_coadd',
                                 image_in_tag='science_coadd',
                                 snr_out_tag='companion_coadd_snr', 
                                 position=pos_pix,
                                 aperture=radius,
                                 output_noise=True)
    pipeline.add_module(module)
    pipeline.run_module('snr_coadd')
    
    snr = pipeline.get_data('companion_coadd_snr')[0]
    data[0, 11] = snr[4] #snr
    data[0, 12] = snr[5] #fpf
    data[0,  4] = snr[6] #companion signal
    data[0,  5] = snr[7] #mean noise
    data[0,  6] = snr[8] #std noise
    
    
    #measure companion stats for each slice
    module = FalsePositiveModule(name_in='snr',
                                 image_in_tag='science_coadd',
                                 snr_out_tag='companion_snr', 
                                 position=pos_pix,
                                 aperture=radius,
                                 output_noise=True)
    pipeline.add_module(module)
    pipeline.run_module('snr')
    
    snr = pipeline.get_data('companion_snr')
    data[1:, 11] = snr[:,4] #snr
    data[1:, 12] = snr[:,5] #fpf
    data[1:,  4] = snr[:,6] #companion signal
    data[1:,  5] = snr[:,7] #mean noise
    data[1:,  6] = snr[:,8] #std noise
    
    
    #calculate mag
    companion_tot = sum(data[3:-2,4])
    try:
        data[0, 13] = -2.5*math.log10(companion_tot/star_tot)
    except:
        print('Error with companion or star flux measurements')
    for i in range(1,len(data)):
        try:
            data[i,13] = -2.5*math.log10(data[i,4]/data[i,1])
        except:
            ...
    
    #save data
    np.savetxt(folder+obs+'_companion'+str(i+1)+'_data.txt', data,
               header='wl, star, avg_noise_star, std_noise_star, comp, '+\
                      'avg_noise_comp, std_noise_comp, sep, sep_err, angle, '+\
                      'angle_err, snr, fpf, d_mag')
