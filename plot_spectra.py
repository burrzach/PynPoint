## Imports ##
import numpy as np
import matplotlib.pyplot as plt


## Settings ##
fit_model = True

obs_list = ["2023-05-27", 
            "2023-05-30-2", 
            "2023-06-15-1",
            "2023-07-26-1",
            "2023-08-07-2"]
companion_list = {"2023-05-27":  2, 
                  "2023-05-30-2":1, 
                  "2023-06-15-1":1,
                  "2023-07-26-1":1,
                  "2023-08-07-2":2}
model = "T4500_g45"

for obs in obs_list:
    n_companions = companion_list[obs]
    
    ## Analysis ##
    #load companion data
    comp_file = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/companions/"+obs+"_companion1_data.txt"
    data = np.genfromtxt(comp_file)
    
    #load stellar model
    star_file = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/companions/starmodel_"+model+".txt"
    star_model = np.genfromtxt(star_file)
    
    
    #print stats
    print(obs)
    print('sep, angle, snr, fpf, mag')
    if n_companions > 1:
        print('companion 1:')
    head = data[0,[7,9,11,12,13]]
    print(head)
    
    
    #format spectra
    wl = data[1:,0] / 1e3 #convert nm -> micron
    wl.sort()
    star_spectra = data[1:,1]
    
    
    #interpolate stellar model
    star_wl = star_model[:,0] / 1e4 #convert angstrom -> micron
    model_spectra = np.interp(wl, star_wl, star_model[:,1])
    ratio = np.mean(star_spectra) / np.mean(model_spectra)
    model_spectra *= ratio
    if fit_model == False:
        model_spectra *= 3
    else:
        #fit star spectrum to model, then use to correct companion spectrum
        fitting = model_spectra / star_spectra
        star_spectra *= fitting
    
    
    #make figure and plot star
    plt.figure('spectra'+obs)
    plt.plot(wl, star_spectra, marker='*', label=obs)
    plt.yscale('log')
    
    #plot first companion
    comp = data[1:,4] - data[1:,6]
    error = data[1:,5]
    snr = data[1:,11]
    
    if fit_model == True:
        comp *= fitting
        error *= fitting
    if n_companions > 1:
        plt.errorbar(wl, comp, yerr=error, marker='o', label='companion 1')
        plt.figure('snr'+obs)
        plt.plot(wl, snr, marker='o', label='companion 1')
    else:
        plt.errorbar(wl, comp, yerr=error, marker='o', label='companion')
        plt.figure('snr'+obs)
        plt.plot(wl, snr, marker='o', label='companion')
        
    
    #repeat for each companion beyond the first
    for i in range(2,n_companions+1):
        #load companion data
        comp_file = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/companions/"+\
            obs+"_companion"+str(i)+"_data.txt"
        data = np.genfromtxt(comp_file)
        
        head = data[0,[7,9,11,12,13]]
        print('companion '+str(i)+':\n', head)
        
        comp = data[1:,4] - data[1:,6]
        error = data[1:,5]
        snr = data[1:,11]
        
        if fit_model == True:
            comp *= fitting
            error *= fitting
        
        plt.figure('spectra'+obs)
        plt.errorbar(wl, comp, yerr=error, marker='o', label='companion '+str(i))
        plt.figure('snr'+obs)
        plt.plot(wl, snr, marker='o', label='companion '+str(i))
    
    
    #plot settings
    plt.figure('spectra'+obs)
    plt.plot(wl, model_spectra, marker='s', label='stellar model')
    plt.legend()
    plt.xlabel('$\lambda$ $[\mu m]$')
    plt.title(obs)
    
    plt.figure('snr'+obs)
    if n_companions > 1:
        plt.legend()
    plt.xlabel('$\lambda$ $[\mu m]$')
    plt.ylabel('SNR [-]')
    plt.ylim(bottom=0)
    plt.title(obs+' snr')
    
    plt.show()
