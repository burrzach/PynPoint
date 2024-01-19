#imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pynpoint import Pypeline, FitsReadingModule, WavelengthReadingModule, \
    ParangReadingModule, DerotateAndStackModule, RemoveFramesModule, \
    AddFramesModule
from pynpoint.util.image import polar_to_cartesian
import matplotlib.animation as animation


#initialize
folder = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/2023-05-30-2/"
companions_file = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/companions/2023-05-30-2_companion1_data.txt"
raw = "SPHER.2023-05-30T01_46_48.038.fits"
dark = "calib/dark_sky_DIT=64.00.fits"
ff = "calib/master_detector_flat_white_l5.fits"
specpos = "calib/spectra_positions.fits"
noise_removed = "preproc/SPHER.2023-05-30T01_46_48.038_DIT000_preproc.fits"
derot_science_cube = "products/science_cube_derotated.fits"
science_cube = "products/science_cube.fits"
residuals = "products/residuals.fits"
coadd_raw = "products/raw_coadd.fits"
star_ob = "products/psf_cube.fits"

wl_file = "products/wavelength.fits"
parang_file = "products/psf_derot.fits"

file = coadd_raw

data_3D = False
draw_circs = True

pipeline = Pypeline(working_place_in=folder,
                    input_place_in=folder,
                    output_place_in=folder)


radius = 0.035
scale = 1.73 / 290
zoom_width = 40

plt.rcParams.update({'font.size': 15})
    
#make subfigure for this observation
plt.figure()
plt.xlabel('RA [arcsec]')
plt.ylabel('Dec [arcsec]')
# plt.tick_params(left = False, right = False , labelleft = False , 
#                 labelbottom = False, bottom = False)

#load raw
module = FitsReadingModule(name_in='read_image',
                           input_dir=None,
                           filenames=[folder+file],
                           image_tag='image',
                           ifs_data=data_3D)
pipeline.add_module(module)
pipeline.run_module('read_image')

if data_3D:
    module = WavelengthReadingModule(name_in='wavelength',
                                         data_tag='image',
                                         file_name=folder+wl_file)
    pipeline.add_module(module)
    pipeline.run_module('wavelength')
    
    module = ParangReadingModule(name_in='parang',
                                 data_tag='image', 
                                 file_name=folder+parang_file)
    pipeline.add_module(module)
    pipeline.run_module('parang')
    
    wl = pipeline.get_attribute(data_tag='image', 
                                attr_name='WAVELENGTH',
                                static=False)
    wl = np.round(wl/1e3, 2)
    
    
    module = DerotateAndStackModule(name_in='derotate_psf',
                                    image_in_tag='image',
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
                             image_out_tag='image')
    pipeline.add_module(module)
    pipeline.run_module('coadd_psf')

image = pipeline.get_data('image')
#data_3D = False

#make subfigure
#plot raw
size = image.shape[-1] * scale / 2

if data_3D:
    image = image.reshape((39,290,290))
    
mins = []
maxs = []
for im in image:
    mins.append(np.quantile(im, 0.01))
    maxs.append(np.quantile(im, 0.99))
lb = np.mean(mins)
ub = np.mean(maxs)

norm = colors.Normalize(vmin=lb, vmax=ub, clip=True)

#for all normal figures
img = plt.imshow(image[0], origin='lower', extent=[size, -size, -size, size],
                  norm=norm)
cb = plt.colorbar(img, location='left', shrink=0.5)
cb.set_label('Flux [counts]', size=15.)


#for animation of 3D cube
if data_3D == True:
    plt.title(r'$\lambda$='+str(wl[19])+r' $\mu$m') #afix title to single slice
    
    fig = plt.figure()
    plt.xlabel('RA [arcsec]')
    plt.ylabel('Dec [arcsec]')
    im = plt.imshow(image[0], animated=True, norm=norm, origin='lower',
                    extent = [size, -size, -size, size])
    cb = plt.colorbar(im, location='left', shrink=0.5, pad=0.2)
    cb.set_label('Flux [counts]', size=15.)
    wl_text = plt.title(r'$\lambda$='+str(wl[0])+r' $\mu$m')
    
    
    def update(i):
        im.set_array(image[i])
        wl_text.set_text(r'$\lambda$='+str(wl[i])+r' $\mu$m')
        return im, wl_text
    
    animation_fig = animation.FuncAnimation(fig, update, frames=len(image), 
                                            interval=200, blit=False, 
                                            repeat_delay=10,)
    
    writer = animation.PillowWriter(fps=5,
                                    bitrate=1800)
    animation_fig.save(folder+'reduced_cube.gif', writer=writer)


#draw circles around image
if draw_circs == True:
    data = np.genfromtxt(companions_file)
    sep = data[0,7]
    angle = data[0,9]
    
    # sep = 3 * radius
    # angle = 0.
    
    n = int(2 * np.pi / np.arcsin(2 * radius / sep))
    theta = 2 * np.pi / n
    
    pos = polar_to_cartesian(image[0], sep/scale, angle)
    y = (pos[0] - image.shape[-1]/2) * scale
    x = (pos[1] - image.shape[-1]/2) * -scale
    
    # circle = plt.Circle((0.,0.), radius=radius, color='orange',
    #                     fill=False, ls=':', lw=2.)
    # plt.gca().add_patch(circle)
    
    circle = plt.Circle((x,y), radius=radius, color='orange',
                        fill=False, ls=':', lw=2.)
    plt.gca().add_patch(circle)
    
    # for i in range(1, n):
    #     x1 = np.cos(theta*i) * x - np.sin(theta*i) * y
    #     y1 = np.sin(theta*i) * x + np.cos(theta*i) * y
        
    #     circle = plt.Circle((x1,y1), radius=radius, color='white',
    #                         fill=False, ls=':', lw=2.)
    #     plt.gca().add_patch(circle)
        