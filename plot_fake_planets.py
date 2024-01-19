# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 16:42:50 2023

@author: Zach
"""

import numpy as np
from pynpoint import Pypeline, FitsReadingModule, ParangReadingModule, WavelengthReadingModule, \
    PSFpreparationModule, AddLinesModule, RemoveFramesModule, AddFramesModule, \
    FakePlanetModule, FalsePositiveModule, PcaPsfSubtractionModule, DerotateAndStackModule, \
    FitsWritingModule
from pynpoint.util.image import polar_to_cartesian
from pynpoint.core.processing import ProcessingModule
#import configparser
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import interp2d

scale = 1.73 / 290  #arcsec/pixel
radius = 0.035      #arcsec
angles = [0., 120., 240.]
sep_list = [0.2, 0.4, 0.6]

folder = '/data/zburr/yses_ifu/2nd_epoch/processed/2023-06-15-2/products/'  #linux
folder = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/2023-06-15-2/" #windows
curve = folder+'contrast_map.txt' #linux
curve = folder+'2023-06-15-2_contrast_map.txt'
psf_folder = folder

plt.rcParams.update({'font.size': 15})

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


#Load science image
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


#Read in contrast curve data
contrast_map = np.genfromtxt(curve)

seps = int((contrast_map.shape[0] / 2))
sep_space = contrast_map[1:seps, 0]
angle_space = contrast_map[0, 1:]

pre_map = contrast_map[1:seps, 1:]
post_map = contrast_map[seps+1:, 1:]

image_in = 'science'
image_out = 'add_1'
count = 1
for i, ang in enumerate(angles):
    for sep in sep_list:
        count += 1
        
        i_ang = np.argmin(abs(angle_space - ang))
        i_sep = np.argmin(abs(sep_space - sep))
        mag = pre_map[i_sep, i_ang]
        print(mag)
        
        module = FakePlanetModule(name_in='fake', 
                                  image_in_tag=image_in, 
                                  psf_in_tag='planet', 
                                  image_out_tag=image_out,
                                  position=(sep, ang), 
                                  magnitude=mag)
        pipeline.add_module(module)
        pipeline.run_module('fake')
        
        image_in = image_out
        image_out = image_out[:-1] + str(count)
        

#coadd image
module = RemoveFramesModule(name_in='slice_science', 
                            image_in_tag=image_in, 
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


#save fits file
module = FitsWritingModule(name_in='write_injected',
                           data_tag='coadd',
                           file_name=folder+'ContrastCurveInjection.fits')
pipeline.add_module(module)
pipeline.run_module('write_injected')


#plot image
raw = pipeline.get_data('coadd')

size = scale * raw.shape[-1]/2.

plt.rcParams.update({'font.size': 15})
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(5.5)
img = plt.imshow(raw[0], origin='lower', extent=[size, -size, -size, size])
cb = plt.colorbar(img, location='right', shrink=0.8)
cb.set_label('Flux [counts]')

plt.xlabel('RA [arcsec]')
plt.ylabel('Dec [arcsec]')

plt.savefig(folder+'ContrastCurveInjection.png', bbox_inches='tight')



#%%
#scatter plot
plt.figure()
#colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
for i in range(len(angle_space)):
    plt.scatter(sep_space+np.random.normal(scale=0.002), pre_map[:,i], 
                label=f'{np.round(angle_space[i])} degrees')

plt.xlabel('Separation [arcsec]')
plt.ylabel(r'$\Delta$ mag')

# legend_elements = [Line2D([0], [0], color='w', markerfacecolor='blue', 
#                           marker='^', label='0 deg'),
#                    Line2D([0], [0], color='w', markerfacecolor='orange', 
#                           marker='s', label='120 deg'),
#                    Line2D([0], [0], color='w', markerfacecolor='green', 
#                           marker='o', label='240 deg'),
#                    ]
plt.legend()
plt.gca().invert_yaxis()


#%%
#heat map
x = []
y = []
for ang in angle_space:
    for sep in sep_space:
        sep_pix = sep / scale
        y_pix, x_pix = polar_to_cartesian(raw, sep_pix, ang)
        y.append((y_pix - 290/2) * scale)
        x.append((x_pix - 290/2) * scale)

z_pre = pre_map.flatten('F')
z_post = post_map.flatten('F')

f_pre = interp2d(x, y, pre_map)
f_post = interp2d(x, y, z_post)

y_coords = np.linspace(-1.73/2, 1.73/2, 290)

pre_map_interp = f_pre(y_coords, y_coords)
post_map_interp = f_post(y_coords, y_coords)

# plt.figure()
# plt.imshow(pre_map_interp, extent=[-1.73/2, 1.73/2, -1.73/2, 1.73/2])

# plt.figure()
# plt.imshow(post_map_interp, extent=[-1.73/2, 1.73/2, -1.73/2, 1.73/2])

theta_space = np.linspace(0., 2*np.pi, 7)
radius_space = np.linspace(sep_space[0]-0.025, sep_space[-1]+0.025, len(sep_space)+1)
tr, rr = np.meshgrid(theta_space, radius_space)

xx = rr * np.cos(tr)
yy = rr * np.sin(tr)

plt.figure()
plt.pcolormesh(xx,yy, 1/post_map)


#%%
#2D scatter
plt.figure()
plt.scatter(x,y, s=50)#, c=z_post)

plt.xlabel('RA [arcsec]')
plt.ylabel('Dec [arcsec]')
