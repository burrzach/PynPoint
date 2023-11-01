#imports
import numpy as np
import matplotlib.pyplot as plt


#settings
fit_model = False

obs_list = {"2023-05-27":  'T4750_g50', 
            "2023-05-30-2":'T4000_g45', #for testing, should be same as above
            "2023-07-26-1":'T4250_g30',
            "2023-08-07-2":'T4500_g45'}

f = plt.figure()
#loop through observations
for obs in obs_list.keys():
    #load companion data
    comp_file = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/companions/"+obs+"_companion_data.txt"
    data = np.genfromtxt(comp_file)
    head = data[:7]
    
    #load stellar model
    star_file = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/companions/starmodel_"+obs_list[obs]+".txt"
    star_model = np.genfromtxt(star_file)
    
    #print stats
    print(obs)
    print('sep, sep_err, angle, angle_err, snr, fpf, mag')
    if len(data[0]) > 3:
        for i in range(len(data[0])-2):
            print('companion '+str(i+1)+':\n', head[:,i+2])
    else:
        print(head[:,2])
    
    #format spectra
    spectra = data[7:]
    #spectra = spectra[spectra[:,0].argsort()]
    wl = spectra[:,0] / 1e3 #convert nm -> micron
    wl.sort()
    
    #interpolate stellar model
    star_wl = star_model[:,0] / 1e4 #convert angstrom -> micron
    star_spectra = np.interp(wl, star_wl, star_model[:,1])
    ratio = np.mean(spectra[:,1]) / np.mean(star_spectra)
    star_spectra *= ratio
    star_spectra *= 3
    
    #make figure and plot star
    # plt.figure()
    plt.plot(wl, spectra[:,1], marker='o', label=obs)
    if obs == '2023-08-07-2':
        plt.plot(wl, star_spectra, marker='s', label=obs_list[obs])
    # plt.yscale('log')
    
    # #plot companions
    # for i in range(len(data[0])-2):
    #     comp = spectra[:,i+2]
    #     if len(data[0]) > 3:
    #             plt.plot(wl, comp, marker='o', label='companion '+str(i+1))
    #     else:
    #         plt.plot(wl, comp, marker='o', label='companion')
    
    # plt.legend()
    # plt.xlabel('$\lambda$ $[\mu m]$')
    # plt.title(obs)
    # plt.show()
    
    # plt.figure(f)
    # plt.plot(wl, star_spectra, marker='o', label=obs_list[obs])
plt.xlabel('$\lambda$ $[\mu m]$')
plt.legend()
plt.show()
