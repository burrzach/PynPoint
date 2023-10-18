#imports
import numpy as np
import math
from pynpoint import Pypeline, FitsReadingModule, ParangReadingModule, WavelengthReadingModule,\
    AddFramesModule, RemoveFramesModule, FalsePositiveModule, AperturePhotometryModule, \
    DerotateAndStackModule
from pynpoint.core.processing import ProcessingModule


#Settings
folder = "/data/zburr/yses_ifu/2nd_epoch/processed/2023-05-27/products/"
pos_guess = (247., 146.)


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


## Coadd images ##
#coadd science
module = DerotateAndStackModule(name_in='derotate_science',
                                image_in_tag='science',
                                image_out_tag='science_derot',
                                derotate=True,
                                stack=None)
pipeline.add_module(module)
pipeline.run_module('derotate_science')

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

#coadd psf
module = DerotateAndStackModule(name_in='derotate_psf',
                                image_in_tag='psf',
                                image_out_tag='psf_derot',
                                derotate=True,
                                stack=None)
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


## Find position ##
module = FalsePositiveModule(name_in='find_companion',
                             image_in_tag='science_coadd',
                             snr_out_tag='companion_snr', 
                             position=pos_guess, 
                             optimize=True,
                             tolerance=0.01,
                             offset=50)
pipeline.add_module(module)
pipeline.run_module('find_companion')

snr = pipeline.get_data('companion_snr')[0]
print('x position (pix), y position (pix), separation (arcsec), position angle (deg), SNR, FPF')
print(snr)
pos_pix = (snr[0], snr[1])
sep = snr[2]
angle = snr[3]


## Measure spectra ##
spectra = np.zeros((39, 3))
spectra[:,0] = pipeline.get_attribute('science', 'WAVELENGTH', static=False)

#measure companion spectrum
module = ReshapeModule(name_in='shape_down_science', 
                       image_in_tag='science_derot', 
                       image_out_tag='science3D', 
                       shape=(39,290,290))
pipeline.add_module(module)
pipeline.run_module('shape_down_science')

module = AperturePhotometryModule(name_in='measure_companion', 
                                  image_in_tag='science3D', 
                                  phot_out_tag='companion_phot',
                                  radius=0.15,
                                  position=pos_pix)
pipeline.add_module(module)
pipeline.run_module('measure_companion')

phot = pipeline.get_data('companion_phot')
print(phot)

#measure star spectrum
module = ReshapeModule(name_in='shape_down_psf', 
                       image_in_tag='psf_derot', 
                       image_out_tag='psf3D', 
                       shape=(39,80,80))
pipeline.add_module(module)
pipeline.run_module('shape_down_psf')

module = AperturePhotometryModule(name_in='measure_star', 
                                  image_in_tag='psf3D', 
                                  phot_out_tag='star_phot',
                                  radius=0.15,
                                  position=None)
pipeline.add_module(module)
pipeline.run_module('measure_star')

spectra[:,2] = pipeline.get_data('star_phot')


## Output data ##
companion_tot = sum(spectra[2:-2,1])
star_tot = sum(spectra[2:-2,2])
mag = -2.5*math.log10(companion_tot/star_tot)

data = np.array([sep, angle, mag])
data = np.vstack((data, spectra))

np.savetxt(folder+'companion_data.txt', data)
