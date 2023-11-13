#imports
import numpy as np
import matplotlib.pyplot as plt

#settings
folder = "D:/Zach/Documents/TUDelft/MSc/Thesis/PynPoint/6-15-2/"
angle_step = 60     #deg
sep_step = 0.05     #arcsec
inner_radius = 0.15 #arcsec
outer_radius = 0.8  #arcsec

#load data
pre_map = np.genfromtxt(folder + "contrast_map_pre.txt")
post_map = np.genfromtxt(folder + "contrast_map_post.txt")

#calculations
sep_space = np.arange(inner_radius, outer_radius, sep_step)
angle_space = np.arange(0., 360., angle_step)

pre_curve = [np.mean(pre_map[i]) for i in range(len(pre_map))]
post_curve = [np.mean(post_map[i]) for i in range(len(post_map))]


#plot
plt.figure()
plt.plot(sep_space, pre_curve, marker='o', label='Before SDI')
plt.plot(sep_space, post_curve, marker='o', label='After SDI')
plt.gca().invert_yaxis()
plt.xlim(xmin=0)
plt.xlabel('Separation [arcsec]')
plt.ylabel('Contrast [-]')
plt.legend()
plt.show()
