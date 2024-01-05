#imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pynpoint import Pypeline, FitsReadingModule
from pynpoint.util.image import polar_to_cartesian
import pandas as pd


#initialize
folder = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/cubes/"
props_file = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/star_data.csv"
output_folder = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/cubes/plotted/"
companions_folder = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/companions/"

pipeline = Pypeline(working_place_in=folder,
                    input_place_in=folder,
                    output_place_in=folder)

#settings
star_props = pd.read_csv(props_file, index_col=0)
star_props = star_props.drop_duplicates(subset='2MASS', keep='last')
observations = np.array(star_props['obs'])
# observations = np.array(["2023-05-27",    #just systems with candidates
#                          "2023-05-30-2", 
#                          "2023-06-15-1",
#                          "2023-07-26-1",
#                          "2023-08-07-2"
#                          ])

companion_list = {"2023-05-27":  2, #how many companions are in each system
                  "2023-05-30-2":1, 
                  "2023-06-15-1":1,
                  "2023-07-26-1":1,
                  "2023-08-07-2":2}
radius = 0.035
bi_radius = 0.09
scale = 1.73 / 290
zoom_width = 40

plt.rcParams.update({'font.size': 15})
    
#loop through observations in this subset
for ob in observations:
    name = star_props.loc[star_props['obs'] == ob, '2MASS'].iloc[0]

    #make subfigure for this observation
    if ob in companion_list.keys():
        fig, ag = plt.subplots(nrows=2, ncols=2, sharey='row', layout='constrained', num=f'{ob}_coadd_grid')
        ag = ag.reshape((4))
        fig.set_figheight(10)
        fig.set_figwidth(11)
        
    else:
        fig, ag = plt.subplots(nrows=1, ncols=2, sharey=True, layout='constrained', num=f'{ob}_coadd_grid')
        fig.set_figheight(5)
        fig.set_figwidth(11)
    
    fig.suptitle(f'{ob}: 2MASS J{name}')
    ag[0].set_title('Before subtraction')
    ag[0].set_ylabel('Dec [arcsec]')
    ag[0].set_xlabel('RA [arcsec]')
    ag[1].set_title('After subtraction')
    ag[1].set_xlabel('RA [arcsec]')
    
    #load raw
    try:
        module = FitsReadingModule(name_in='read_raw',
                                   input_dir=None,
                                   filenames=[folder+'raw_coadd/'+ob+"_raw_coadd.fits"],
                                   image_tag='raw')
        pipeline.add_module(module)
        pipeline.run_module('read_raw')
        
        #load resid
        module = FitsReadingModule(name_in='read_resid',
                                   input_dir=None,
                                   filenames=[folder+'resid_coadd/'+ob+"_resid_coadd.fits"],
                                   image_tag='resid')
        pipeline.add_module(module)
        pipeline.run_module('read_resid')
        
        raw = pipeline.get_data('raw')
        resid = pipeline.get_data('resid')
                
    except:
        raw = np.zeros((1,290,290))
        resid = np.zeros((1,290,290))
    
    #make subfigure
    #plot raw
    size = scale * raw.shape[-1]/2.
    
    img = ag[0].imshow(raw[0], origin='lower', extent=[size, -size, -size, size])
    if ob in companion_list.keys():
        cb = plt.colorbar(img, ax=[ag[0], ag[2]], location='left', shrink=0.5)
    else:
        cb = plt.colorbar(img, ax=ag[0], location='left', shrink=0.5)
    cb.set_label('Flux [counts]', size=15.)
    
    #plot resid
    size = scale * resid.shape[-1]/2.
    if ob in companion_list.keys():
        img = ag[1].imshow(resid[0], origin='lower', extent=[size, -size, -size, size])
        cb = plt.colorbar(img, ax=[ag[1], ag[3]], location='right', shrink=0.5)
    else:
        img = ag[1].imshow(resid[0], origin='lower', extent=[size, -size, -size, size],
                           norm=colors.Normalize(vmin=None, vmax=np.max(resid[0])*0.8, clip=True))
        cb = plt.colorbar(img, ax=ag[1], location='right', extend='max', shrink=0.5)
    cb.set_label('Flux [counts]', size=15.)
    
    if ob in companion_list.keys():
        if ob == '2023-06-15-1':
            app_rad = bi_radius
        else:
            app_rad = radius
        n_companions = companion_list[ob]
        comp_pos = np.zeros((n_companions,2))
        
        #circle companions
        circles = [] #!!!
        for j in range(1, n_companions+1):
            comp_file = companions_folder + ob + f"_companion{j}_data.txt"
            data = np.genfromtxt(comp_file)
            
            sep = data[0,7]
            angle = data[0,9]
            pos = polar_to_cartesian(raw[0], sep/scale, angle)
            y = (pos[0] - raw.shape[-1]/2) * scale
            x = (pos[1] - raw.shape[-1]/2) * -scale
            
            circle0 = plt.Circle((x,y), radius=app_rad, color='red',
                                fill=False, ls=':', lw=2.)
            circles.append(circle0) #!!!
            # ag[0].add_patch(circle0)
            
            # circle1 = plt.Circle((x,y), radius=app_rad, color='red',
            #                     fill=False, ls=':', lw=2.) #need 2 different circles else plt throws error
            # ag[1].add_patch(circle1)
         
        
            comp_pos[j-1] = np.round(pos)
        
        #zoom in on companions
        zw = int(zoom_width/2)
        y_comp = int(np.mean(comp_pos[:,0]))
        x_comp = int(np.mean(comp_pos[:,1]))
        raw_zoom = raw[0][y_comp-zw:y_comp+zw, x_comp-zw:x_comp+zw]
        resid_zoom = resid[0][y_comp-zw:y_comp+zw, x_comp-zw:x_comp+zw]
        
        center = raw.shape[-1]/2.
        shift = ((center-x_comp) * scale, (y_comp-center) * scale)
        size = scale * raw_zoom.shape[-1]/2.
        extent = [shift[0]+size, shift[0]-size, shift[1]-size, shift[1]+size]
        
        square0 = plt.Rectangle((extent[1],extent[2]), width=2*size, height=2*size,
                               color='grey', fill=False, ls='--', lw=3.)
        ag[0].add_patch(square0)
        square1 = plt.Rectangle((extent[1],extent[2]), width=2*size, height=2*size,
                               color='grey', fill=False, ls='--', lw=3.)
        ag[1].add_patch(square1)
        
        #plt.figure()
        ag[2].set_title('Closeup on companions')
        ag[2].imshow(raw_zoom, origin='lower', extent=extent,
                   norm=colors.Normalize(vmin=np.min(raw[0]), vmax=np.max(raw[0])))
        
        ag[3].set_title('Closeup on companions')
        ag[3].imshow(resid_zoom, origin= 'lower', extent=extent,
                     norm=colors.Normalize(vmin=np.min(resid[0]), vmax=np.max(resid[0])))
        
        ag[2].set_ylabel('Dec [arcsec]')
        ag[2].set_xlabel('RA [arcsec]')
        ag[3].set_xlabel('RA [arcsec]')
    
    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()
    # plt.show()
    # plt.savefig(output_folder + f'{ob}_coadd_grid.png')
    # plt.close()
    
    # plt.figure()
    # plt.imshow(raw_zoom, origin='lower', extent=extent,
    #            norm=colors.Normalize(vmin=np.min(raw[0]), vmax=np.max(raw[0])))
    # for circle in circles:
    #     plt.gca().add_patch(circle)
    # plt.ylabel('Dec [arcsec]')
    # plt.xlabel('RA [arcsec]')
    
    # plt.savefig(output_folder + f'aperture_{ob}.png')
    # plt.close()
