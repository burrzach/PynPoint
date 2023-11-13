#imports
import numpy as np
import matplotlib.pyplot as plt

#settings
folder = "D:/Zach/Documents/TUDelft/MSc/Thesis/PynPoint/6-15-2/"

#load data
pre_map = np.genfromtxt(folder + "contrast_map_pre.txt")
post_map = np.genfromtxt(folder + "contrast_map_post.txt")

#calculations
sep_space = pre_map[1:, 0]
angle_space = pre_map[0, 1:]

pre_map = pre_map[1:, 1:]
post_map = post_map[1:, 1:]

pre_curve = [np.mean(pre_map[i]) for i in range(len(pre_map))]
pre_error = [np.std(pre_map[i]) for i in range(len(pre_map))]
post_curve = [np.mean(post_map[i]) for i in range(len(post_map))]
post_error = [np.std(post_map[i]) for i in range(len(post_map))]


#plot
plt.figure()
plt.errorbar(sep_space, pre_curve, yerr=pre_error, marker='o', capsize=3, label='Before SDI')
plt.errorbar(sep_space, post_curve, yerr=post_error, marker='o', capsize=3, label='After SDI')
plt.gca().invert_yaxis()
plt.xlim(xmin=0)
plt.xlabel('Separation [arcsec]')
plt.ylabel('Contrast [-]')
plt.legend()
plt.show()
