# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 12:47:26 2024

@author: Zach
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pynpoint import Pypeline, FitsReadingModule, DerotateAndStackModule,\
    ParangReadingModule, WavelengthReadingModule, PSFpreparationModule,\
    AddLinesModule, RemoveFramesModule, AddFramesModule, FakePlanetModule
from pynpoint.util.sdi import sdi_scaling, scaling_factors
from pynpoint.core.processing import ProcessingModule
import configparser
from scipy.interpolate import interp2d



folder = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/2023-05-30-2/products/"
angle = 295.
radius = 0.035
scale = 1.73 / 290
mag = 7.
sep1 = 0.35
sep2 = 0.45


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


#Derotate image
module = DerotateAndStackModule(name_in='rotate', 
                                image_in_tag='science', 
                                image_out_tag='science_derot',
                                extra_rot=angle)
pipeline.add_module(module)
pipeline.run_module('rotate')

pipeline.set_attribute(data_tag='science_derot', 
                       attr_name='PARANG', 
                       attr_value=np.array([0.]),
                       static=False)

#Perform SDI



## Load PSF ##
# module = FitsReadingModule(name_in='readpsf',
#                             image_tag='psf',
#                             filenames=[folder+'psf_cube.fits'],
#                             input_dir=None,
#                             ifs_data=True)
# pipeline.add_module(module)
# pipeline.run_module('readpsf')

# module = ParangReadingModule(name_in='parangpsf',
#                               data_tag='psf',
#                               file_name=folder+'psf_derot.fits')
# pipeline.add_module(module)
# pipeline.run_module('parangpsf')

# module = WavelengthReadingModule(name_in='wavelengthpsf',
#                                   data_tag='psf',
#                                   file_name=folder+'wavelength.fits')
# pipeline.add_module(module)
# pipeline.run_module('wavelengthpsf')


# ## Prepare PSF for injection ##
# module = RemoveFramesModule(name_in='slice_psf', 
#                             image_in_tag='psf', 
#                             selected_out_tag='other_psf', 
#                             removed_out_tag='psf_slice', 
#                             frames=[20])
# pipeline.add_module(module)
# pipeline.run_module('slice_psf')

# module = PSFpreparationModule(name_in='maskpsf', 
#                               image_in_tag='psf_slice', 
#                               image_out_tag='psf_masked',
#                               cent_size=None,
#                               edge_size=radius)
# pipeline.add_module(module)
# pipeline.run_module('maskpsf')

# module = AddLinesModule(name_in='pad', 
#                         image_in_tag='psf_masked', 
#                         image_out_tag='psf_pad', 
#                         lines=(105,105,105,105))
# pipeline.add_module(module)
# pipeline.run_module('pad')

# module = ReshapeModule(name_in='reshape_psf', 
#                        image_in_tag='psf_pad', 
#                        image_out_tag='planet', 
#                        shape=(1,1,290,290))
# pipeline.add_module(module)
# pipeline.run_module('reshape_psf')


# #inject two fake planets
# module = FakePlanetModule(name_in='fake1',
#                           image_in_tag='science_derot', 
#                           psf_in_tag='planet', 
#                           image_out_tag='injected1', 
#                           position=(sep1,0.), 
#                           magnitude=mag)
# pipeline.add_module(module)
# pipeline.run_module('fake1')

# module = FakePlanetModule(name_in='fake2',
#                           image_in_tag='injected1', 
#                           psf_in_tag='planet', 
#                           image_out_tag='injected2', 
#                           position=(sep2,0.), 
#                           magnitude=mag)
# pipeline.add_module(module)
# pipeline.run_module('fake2')



#grab data
raw = pipeline.get_data('science_derot')
raw = raw.reshape((39,290,290))

#plt.imshow(raw[20])


#Rescale image
wl = pipeline.get_attribute(data_tag='science_derot', 
                            attr_name='WAVELENGTH',
                            static=False)
x_coords = np.linspace(min(wl),max(wl),39)
scales = scaling_factors(x_coords)

scaled = sdi_scaling(raw, scales)


#Take median
med = np.median(scaled, axis=0)

#subtract
subtracted = scaled - med

#rescale
rescaled = sdi_scaling(subtracted, 1/scales, reverse=True)


#slice
x = int(290/2)

lam_y = raw[:,:,x]
lam_y_scaled = scaled[:,:,x]

lam_y_sub = subtracted[:,:,x]
lam_y_rescale = rescaled[:,:,x]

#lam_y[:,x-25:x+25] = np.zeros((39,50))
#lam_y_scaled[:,x-25:x+25] = np.zeros((39,50))


#interpolate
y = np.linspace(-1.73/2, 1.73/2, 290)

f = interp2d(y, wl, lam_y)
Z = f(y, x_coords)

f_scaled = interp2d(y, wl, lam_y_scaled)
Z_scaled = f_scaled(y, x_coords)

f_sub = interp2d(y, wl, lam_y_sub)
Z_sub = f_sub(y, x_coords)

f_rescale = interp2d(y, wl, lam_y_rescale)
Z_rescale = f_rescale(y, x_coords)

#plot
norm = colors.Normalize(vmin=np.quantile(Z, 0.01), 
                        vmax=np.quantile(Z, 0.99), 
                        clip=True)

norm_sub = colors.Normalize(vmin=np.quantile(Z_sub, 0.01), 
                            vmax=np.quantile(Z_sub, 0.99), 
                            clip=True)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=False, constrained_layout=True)
img = ax1.imshow(Z, extent=[min(y),max(y),min(wl),max(wl)], origin="lower", aspect=1/1000,
           norm=norm)
ax2.imshow(Z_scaled, extent=[min(y),max(y),min(wl),max(wl)], origin="lower", aspect=1/1000,
           norm=norm)

ax1.set_title('Reduced Image')
ax2.set_title('Scaled by Wavelength')
ax1.set_ylabel(r'$\lambda$ $[\mu m]$')
ax2.set_ylabel(r'$\lambda$ $[\mu m]$')
ax1.set_xlabel(r'Separation [arcsec]')
ax2.set_xlabel(r'$\frac{\lambda}{D}$')
cb = plt.colorbar(img, ax=(ax1,ax2), location='right', shrink=0.5)
cb.set_label('Flux [counts]')

img = ax3.imshow(Z_sub, extent=[min(y),max(y),min(wl),max(wl)], origin="lower", 
                 aspect=1/1000, norm=norm_sub)
ax4.imshow(Z_rescale, extent=[min(y),max(y),min(wl),max(wl)], origin="lower", 
           aspect=1/1000, norm=norm_sub)

ax3.set_title('After Subtraction')
ax4.set_title('Rescaled to Original Size')
ax3.set_ylabel(r'$\lambda$ $[\mu m]$')
ax4.set_ylabel(r'$\lambda$ $[\mu m]$')
ax4.set_xlabel(r'Separation [arcsec]')
ax3.set_xlabel(r'$\frac{\lambda}{D}$')
cb = plt.colorbar(img, ax=(ax3,ax4), location='right', shrink=0.5)
cb.set_label('Flux [counts]')
