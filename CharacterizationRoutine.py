#imports
import numpy as np
import math
from pynpoint import Pypeline, FitsReadingModule, ParangReadingModule, WavelengthReadingModule,\
    AddFramesModule, RemoveFramesModule, FalsePositiveModule, AperturePhotometryModule, \
    DerotateAndStackModule, FitCenterModule, FakePlanetModule, PSFpreparationModule, \
    AddLinesModule, FitsWritingModule
from pynpoint.core.processing import ProcessingModule
from pynpoint.util.image import cartesian_to_polar, center_subpixel
import configparser


#Settings
obs = '2023-08-07-2'
#pos_guess = [(247., 146.), (253., 162.)] #2023-05-27
#pos_guess = [(211.5, 176.5)] #2023-05-30-2
#pos_guess = [(109., 58.)] #2023-07-26-1
pos_guess = [(213.,90.), (224.,85.)] #2023-08-07-2
radius = 0.025
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
'''
module = DerotateAndStackModule(name_in='derotate_psf',
                                image_in_tag='psf',
                                image_out_tag='psf_derot',
                                derotate=True,
                                stack=None)
pipeline.add_module(module)
pipeline.run_module('derotate_psf')
'''

module = DerotateAndStackModule(name_in='derotate_psf', #!!! only for 08-07-2, where psf has 3time slices
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
                              edge_size=radius)
pipeline.add_module(module)
pipeline.run_module('maskpsf')

module = AddLinesModule(name_in='pad', 
                        image_in_tag='psf_masked', 
                        image_out_tag='planet', 
                        lines=(105,105,105,105))
pipeline.add_module(module)
pipeline.run_module('pad')


## Measure star spectrum ##
spectra = np.zeros((39, 2+len(pos_guess)))
spectra[:,0] = pipeline.get_attribute('psf', 'WAVELENGTH', static=False)

module = AperturePhotometryModule(name_in='measure_star', 
                                  image_in_tag='psf3D', 
                                  phot_out_tag='star_phot',
                                  radius=0.035,
                                  position=None)
pipeline.add_module(module)
pipeline.run_module('measure_star')

spectra[:,1] = pipeline.get_data('star_phot')[:,0]
star_tot = sum(spectra[2:-2,1])


pic = pipeline.get_data('science_coadd')
center = center_subpixel(pic)


## Loop for multiple companions ##
data = np.full((6,2+len(pos_guess)), np.nan)
for i, guess in enumerate(pos_guess):
    #remove all other planets
    science_image = 'science_coadd'
    image_out = 'removed1'
    count = 1
    for j, guess2 in enumerate(pos_guess):
        if j != i:
            count += 1
            inject_pos = cartesian_to_polar(center, guess2[1], guess2[0])
            inject_pos = (inject_pos[0]*scale, inject_pos[1])

            module = FakePlanetModule(name_in='fake',
                                      image_in_tag=science_image, 
                                      psf_in_tag='planet', 
                                      image_out_tag=image_out, 
                                      position=inject_pos, 
                                      magnitude=1.5,
                                      psf_scaling=-1.)
            pipeline.add_module(module)
            pipeline.run_module('fake')
            
            science_image = image_out
            image_out = image_out[:-1] + str(count)
            
            module = FitsWritingModule(name_in='write_removed', 
                                       data_tag=science_image, 
                                       file_name=folder+obs+'_'+science_image+'.fits')
            pipeline.add_module(module)
            pipeline.run_module('write_removed')
            
            
    #find planet position
    module = FitCenterModule(name_in='fit',
                             image_in_tag=science_image,
                             fit_out_tag='companion_pos',
                             mask_radii=(0.1,1.2),
                             sign='positive',
                             model='gaussian',
                             filter_size=None)
    pipeline.add_module(module)
    pipeline.run_module('fit')

    comp_fit = pipeline.get_data('companion_pos')[0]
    pos_pix = (comp_fit[0]+center[0], comp_fit[2]+center[1])
    pos_pol = np.array(cartesian_to_polar(center, pos_pix[1], pos_pix[0]))
    pos_pol[0] *= scale

    err1 = np.array(cartesian_to_polar(center, pos_pix[1]-comp_fit[3], pos_pix[0]-comp_fit[1]))
    err2 = np.array(cartesian_to_polar(center, pos_pix[1]+comp_fit[3], pos_pix[0]+comp_fit[1]))
    pos_err = abs(err1 - err2)
    
    data[0, i+2] = pos_pol[0] #sep
    data[1, i+2] = pos_err[0] #sep error
    data[2, i+2] = pos_pol[1] #angle
    data[3, i+2] = pos_err[1] #angle error
    
    print('Companion '+str(i)+' position (pix, sep/angle): ', pos_pix, pos_pol)
    
    #measure snr
    module = FalsePositiveModule(name_in='find_companion',
                                 image_in_tag='science_coadd',
                                 snr_out_tag='companion_snr', 
                                 position=pos_pix,
                                 aperture=radius)
    pipeline.add_module(module)
    pipeline.run_module('find_companion')
    
    snr = pipeline.get_data('companion_snr')[0]
    data[4, i+2] = snr[4] #snr
    data[5, i+2] = snr[5] #fpf
    
    
    #measure companion spectrum
    module = AperturePhotometryModule(name_in='measure_companion', 
                                      image_in_tag='science3D', 
                                      phot_out_tag='companion_phot',
                                      radius=radius,
                                      position=pos_pix)
    pipeline.add_module(module)
    pipeline.run_module('measure_companion')
    
    spectra[:,i+2] = pipeline.get_data('companion_phot')[:,0]
    
    
    #calculate mag
    companion_tot = sum(spectra[2:-2,i+2])
    try:
        data[2, i+2] = -2.5*math.log10(companion_tot/star_tot)
    except:
        print('Error with companion or star flux measurements')


## Format and output data ##
data = np.vstack((data,spectra))
np.savetxt(folder+obs+'_companion_data.txt', data, 
           header='# columns: wl,star,companions;\n# rows: sep,sep_err,angle,angle_err,snr,fpf,d_mag')

# module = TextWritingModule(name_in='write_centering', 
#                            data_tag='science_centering', 
#                            file_name=folder+obs+'_centering_data.txt',
#                            header='#x offset (pix), x offset err (pix),'+\
#                                   ' y offset (pix), y offset err (pix),'+\
#                                   ' FWHM major axis (arcsec), FWHM major axis err (arcsec),'+\
#                                   ' FWHM minor axis (arcsec), FWHM minor axis err (arcsec),'+\
#                                   ' amp (ADU), amp err (ADU), angle (deg), angle err (deg),'+\
#                                   ' offset (ADU), offset err (ADU)')
# pipeline.add_module(module)
# pipeline.run_module('write_centering')
