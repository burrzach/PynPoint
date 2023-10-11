#imports
import configparser
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle

from pynpoint import Pypeline, FitsReadingModule, ParangReadingModule, \
                     StarExtractionModule, BadPixelSigmaFilterModule, \
                     StarAlignmentModule, FitCenterModule, ShiftImagesModule, \
                     PSFpreparationModule, PcaPsfSubtractionModule, \
                     FalsePositiveModule, SimplexMinimizationModule, \
                     FakePlanetModule, ContrastCurveModule, SDIContrastCurveModule, \
                     FitsWritingModule, TextWritingModule


#download data
#urllib.request.urlretrieve('https://home.strw.leidenuniv.nl/~stolker/pynpoint/hd142527_zimpol_h-alpha.tgz',
#                           'hd142527_zimpol_h-alpha.tgz')
#tar = tarfile.open('hd142527_zimpol_h-alpha.tgz')
#tar.extractall(path='input')


#initialize
pipeline = Pypeline(working_place_in='./',
                    input_place_in='input/',
                    output_place_in='./')

#read in data
module = FitsReadingModule(name_in='read',
                           input_dir=None,
                           image_tag='zimpol',
                           overwrite=True,
                           check=False,
                           filenames=None,
                           ifs_data=False)
pipeline.add_module(module)
pipeline.run_module('read')

module = ParangReadingModule(name_in='parang',
                             data_tag='zimpol',
                             file_name='parang.dat',
                             input_dir=None,
                             overwrite=True)
pipeline.add_module(module)
pipeline.run_module('parang')

print(pipeline.get_attribute('zimpol', 'PARANG', static=False))

#filter bad pixels
module = BadPixelSigmaFilterModule(name_in='badpixel',
                                   image_in_tag='zimpol',
                                   image_out_tag='bad',
                                   map_out_tag=None,
                                   box=9,
                                   sigma=3.,
                                   iterate=3)
pipeline.add_module(module)
pipeline.run_module('badpixel')

#center image
module = StarExtractionModule(name_in='extract',
                              image_in_tag='bad',
                              image_out_tag='crop',
                              index_out_tag=None,
                              image_size=0.2,
                              fwhm_star=0.03,
                              position=(476, 436, 0.1))
pipeline.add_module(module)
pipeline.run_module('extract')

#align images
module = StarAlignmentModule(name_in='align',
                             image_in_tag='crop',
                             ref_image_in_tag=None,
                             image_out_tag='aligned',
                             interpolation='spline',
                             accuracy=10,
                             resize=None,
                             num_references=10,
                             subframe=0.1)
pipeline.add_module(module)
pipeline.run_module('align')

#data = pipeline.get_data('crop')
#plt.imshow(data[0, ], origin='lower')

#fit psf
module = FitCenterModule(name_in='center',
                         image_in_tag='aligned',
                         fit_out_tag='fit',
                         mask_out_tag=None,
                         method='mean',
                         radius=0.1,
                         sign='positive',
                         model='moffat',
                         filter_size=None,
                         guess=(0., 0., 10., 10., 10000., 0., 0., 1.))
pipeline.add_module(module)
pipeline.run_module('center')

#shift images
module = ShiftImagesModule(name_in='shift',
                           image_in_tag='aligned',
                           image_out_tag='centered',
                           shift_xy='fit',
                           interpolation='spline')
pipeline.add_module(module)
pipeline.run_module('shift')

#mask images
module = PSFpreparationModule(name_in='prep1',
                              image_in_tag='centered',
                              image_out_tag='prep',
                              mask_out_tag=None,
                              norm=False,
                              cent_size=0.02,
                              edge_size=0.2)
pipeline.add_module(module)
pipeline.run_module('prep1')

#mask psf
module = PSFpreparationModule(name_in='prep2',
                              image_in_tag='centered',
                              image_out_tag='psf',
                              mask_out_tag=None,
                              norm=False,
                              cent_size=None,
                              edge_size=0.07)
pipeline.add_module(module)
pipeline.run_module('prep2')

#subtraction
module = PcaPsfSubtractionModule(name_in='pca',
                                 images_in_tag='prep',
                                 reference_in_tag='prep',
                                 res_mean_tag='pca_mean',
                                 res_median_tag='pca_median',
                                 basis_out_tag='pca_basis',
                                 pca_numbers=range(1, 31),
                                 extra_rot=-133.,
                                 subtract_mean=True,
                                 processing_type='ADI')
pipeline.add_module(module)
pipeline.run_module('pca')

#measure snr
module = FalsePositiveModule(name_in='snr',
                             image_in_tag='pca_median',
                             snr_out_tag='snr',
                             position=(11., 26.),
                             aperture=5.*0.0036,
                             ignore=True,
                             optimize=False)
pipeline.add_module(module)
pipeline.run_module('snr')

#measure flux+pos
#module = SimplexMinimizationModule(name_in='simplex',
#                                   image_in_tag='centered',
#                                   psf_in_tag='psf',
#                                   res_out_tag='simplex',
#                                   flux_position_tag='fluxpos',
#                                   position=(11, 26),
#                                   magnitude=6.,
#                                   psf_scaling=-1.,
#                                   merit='gaussian',
#                                   aperture=10.*0.0036,
#                                   sigma=0.,
#                                   tolerance=0.01,
#                                   pca_number=range(1, 11),
#                                   cent_size=0.02,
#                                   edge_size=0.2,
#                                   extra_rot=-133.,
#                                   residuals='median',
#                                   reference_in_tag=None,
#                                   offset=None)
#pipeline.add_module(module)
#pipeline.run_module('simplex')

#remove planet
module = FakePlanetModule(name_in='fake',
                          image_in_tag='centered',
                          psf_in_tag='psf',
                          image_out_tag='removed',
                          position=(0.061, 97.3-133.),
                          magnitude=6.1,
                          psf_scaling=-1.,
                          interpolation='spline')
pipeline.add_module(module)
pipeline.run_module('fake')

#compute detection limits w/ original module
module = ContrastCurveModule(name_in='limits',
                             image_in_tag='removed',
                             psf_in_tag='psf',
                             contrast_out_tag='limits',
                             separation=(0.05, 5., 0.01),
                             angle=(0., 360., 60.),
                             threshold=('fpf', 2.87e-7),
                             psf_scaling=1.,
                             aperture=0.02,
                             pca_number=10,
                             cent_size=0.02,
                             edge_size=2.,
                             extra_rot=-133.,
                             residuals='median',
                             snr_inject=100.)
pipeline.add_module(module)
pipeline.run_module('limits')

#w/ new module
module = ContrastCurveModule(name_in='limits2',
                             image_in_tag='removed',
                             psf_in_tag='psf',
                             contrast_out_tag='limits2',
                             separation=(0.05, 5., 0.01),
                             angle=(0., 360., 60.),
                             threshold=('fpf', 2.87e-7),
                             psf_scaling=1.,
                             aperture=0.02,
                             pca_number=10,
                             cent_size=0.02,
                             edge_size=2.,
                             extra_rot=-133.,
                             residuals='median',
                             snr_inject=100.)
pipeline.add_module(module)
pipeline.run_module('limits2')

#write images
module = FitsWritingModule(name_in='write1',
                           data_tag='pca_median',
                           file_name='pca_median.fits',
                           output_dir=None,
                           data_range=None,
                           overwrite=True,
                           subset_size=None)
pipeline.add_module(module)
pipeline.run_module('write1')

#write limits
module = TextWritingModule(name_in='write2',
                           data_tag='limits',
                           file_name='limits.dat',
                           output_dir=None,
                           header='Separation (arcsec) - Contrast (mag) - Variance (mag) - FPF')
pipeline.add_module(module)
pipeline.run_module('write2')

module = TextWritingModule(name_in='write3',
                           data_tag='limits',
                           file_name='limits2.dat',
                           output_dir=None,
                           header='Separation (arcsec) - Contrast (mag) - Variance (mag) - FPF')
pipeline.add_module(module)
pipeline.run_module('write3')
