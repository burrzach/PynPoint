# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 15:48:46 2023

@author: Zach
"""

import pandas as pd
import numpy as np

props_file = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/star_data.csv"
out_file = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/contrast_curves/figures/place_figures.txt"

star_props = pd.read_csv(props_file, index_col=0)
star_props = star_props.drop_duplicates(subset='2MASS', keep='last')
observations = np.array(star_props['obs'])

companion_list = {"2023-05-27":  2, #how many companions are in each system
                  "2023-05-30-2":1, 
                  "2023-06-15-1":1,
                  "2023-07-26-1":1,
                  "2023-08-07-2":2}

with open(out_file, 'a') as f:
    for ob in observations:
        if ob not in companion_list.keys():
            tar = star_props.loc[star_props['obs'] == ob, '2MASS'].iloc[0]
            
            fig_text = \
            "\\begin{figure}[hp]\n"+ \
            "    \\centering\n"+ \
            "    \\includegraphics[width=\\textwidth]{figures/contrast_curves/"+ob+"_contrast_map.png}\n" + \
            "    \\caption"+ \
            "{Contrast curve for the observation of 2MASS J"+tar+".}\n"+ \
            "    \\label{fig:contrast_curve_"+ob+"}\n"+ \
            "\\end{figure}\n\n"
            
            # fig_text = \
            # "\\begin{figure}[hp]\n"+ \
            # "    \\centering\n"+ \
            # "    \\includegraphics[width=\\textwidth]{figures/coadded_cubes/"+ob+"_coadd_grid.png}\n" + \
            # "    \\caption[Coadded cubes for the observation of 2MASS J"+tar+".]"+ \
            # "{Coadded cubes for the observation of 2MASS J"+tar+". The cubes have been derotated "+ \
            # "such that north is up and east is left. The square shape visible in the images "+ \
            # "is the shape of the detector, with part cut off due to the offset of the "+ \
            # "detector with respect to the IFU.}\n"+ \
            # "    \\label{fig:coadd_"+ob+"}\n"+ \
            # "\\end{figure}\n\n"
    
        
            f.write(fig_text)
        