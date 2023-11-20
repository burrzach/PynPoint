## Imports ##
import numpy as np
import matplotlib.pyplot as plt


## Settings ##
fit_model = True     #fit host spectrum to match model
plot_models = True   #plot range of model spectra to compare to companion
plot_host = False    #plot host star spectrum
fit_companion = True #scale models to value of companion

temp_range = range(35, 29, -1) #range of temperatures to plot models
#temp_range = [45]

obs_list = [#"2023-05-27",    #which observations to plot
            #"2023-05-30-2", 
            "2023-06-15-1",
            #"2023-07-26-1",
            "2023-08-07-2"
            ]
companion_list = {"2023-05-27":  2, #how many companions are in each system
                  "2023-05-30-2":1, 
                  "2023-06-15-1":1,
                  "2023-07-26-1":1,
                  "2023-08-07-2":2}
host_temp = {"2023-05-27":  47, #host star temperature/model to compare
             "2023-05-30-2":48, 
             "2023-06-15-1":40,
             "2023-07-26-1":41,
             "2023-08-07-2":45}

folder = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/companions/"

## Define Functions ##
def bin_spectra(model_wl, model_values, central_wl):
    '''
    Bins model spectra to match data.

    Parameters
    ----------
    model_wl : numpy.ndarray
        Wavelengths for the model spectrum that needs to be binned.
    model_values : numpy.ndarray
        Values of the model corresponding to the model_wl.
    central_wl : numpy.ndarray
        1D array of central wavelength values for the bins

    Returns
    -------
    binned_model : numpy.ndarray
        Model spectra binned to match the central_wl.

    '''
    
    #calculate bins centered around central_wl
    bins = np.zeros((len(central_wl)+1))
    bins[1:-1] = (central_wl[1:] + central_wl[:-1]) / 2
    bin_width = np.mean(np.diff(bins[1:-1]))
    bins[0] = bins[1] - bin_width
    bins[-1] = bins[-2] + bin_width
    
    #find which bin each element is in
    lower_bound = model_wl >= bins[0]
    upper_bound = model_wl < bins[-1]
    bounded_wl = model_wl[lower_bound * upper_bound]
    bin_indices = np.digitize(bounded_wl, bins)
    
    #calculate bin average
    bounded_model = model_values[lower_bound * upper_bound]
    binned_model = [bounded_model[bin_indices == i].mean() \
                    for i in range(1, len(bins))]
    
    return np.array(binned_model)


for obs in obs_list:
    n_companions = companion_list[obs]
    
    ## Analysis ##
    #load companion data
    comp_file = folder+obs+"_companion1_data.txt"
    data = np.genfromtxt(comp_file)
    
    
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
    star_spectra = data[1:,1] #- data[1:,2]
    star_error = data[1:,3]
    
    #load stellar model
    star_file = folder+"star_models/BT_0"+str(host_temp[obs])+"_45_0_0.txt"
    star_model = np.genfromtxt(star_file)
    
    #bin stellar model
    star_wl = star_model[:,0] / 1e4 #convert angstrom -> micron
    model_spectra = bin_spectra(star_wl, star_model[:,1], wl)
    ratio = np.mean(star_spectra) / np.mean(model_spectra)
    model_spectra *= ratio
    if fit_model == False:
        model_spectra *= 3
    else:
        #fit star spectrum to model, then use to correct companion spectrum
        fitting = model_spectra / star_spectra
        star_spectra *= fitting
        star_error *= fitting
    
    
    #make figure and plot star
    plt.figure('spectra'+obs)
    plt.yscale('log')
    if plot_host:
        plt.errorbar(wl, star_spectra, yerr=star_error, marker='*', 
                     label='host star (T_eff='+str(host_temp[obs]*100)+')')
    
    #plot first companion
    comp = data[1:,4] - data[1:,6]
    error = data[1:,5]
    snr = data[1:,11]
    
    if obs == "2023-06-15-1":
        comp *= 0.5 #divide by 2 because companion is a binary
    
    if fit_model == True:
        comp *= fitting
        error *= fitting
    if n_companions > 1:
        plt.errorbar(wl, comp, yerr=error, marker='o', label='companion 1', color='orange')
        #plt.figure('snr'+obs)
        #plt.plot(wl, snr, marker='o', label='companion 1')
    else:
        plt.errorbar(wl, comp, yerr=error, marker='o', label='companion', color='orange')
        #plt.figure('snr'+obs)
        #plt.plot(wl, snr, marker='o', label='companion')
        
    
    #determine scaling for models if necessary
    if fit_companion:
        #load stellar model
        temp = int(np.round(np.mean(temp_range),0))
        star_file = folder+"star_models/BT_0"+str(temp)+"_45_0_0.txt"
        star_model = np.genfromtxt(star_file)
        
        #bin stellar model
        star_wl = star_model[:,0] / 1e4 #convert angstrom -> micron
        model_spectra = bin_spectra(star_wl, star_model[:,1], wl)
        
        #calculate ratio
        ratio = np.mean(comp) / np.mean(model_spectra)
        if n_companions > 1:
            ratio *= 0.8
        
    
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
        plt.errorbar(wl, comp, yerr=error, marker='o', label='companion '+str(i), color='green')
        #plt.figure('snr'+obs)
        #plt.plot(wl, snr, marker='o', label='companion '+str(i))
        
    
    #plot each model
    if plot_models:
        plt.figure('spectra'+obs)
        for temp in temp_range:
            #load stellar model
            star_file = folder+"star_models/BT_0"+str(temp)+"_45_0_0.txt"
            star_model = np.genfromtxt(star_file)
            
            #bin stellar model
            star_wl = star_model[:,0] / 1e4 #convert angstrom -> micron
            model_spectra = bin_spectra(star_wl, star_model[:,1], wl)
            model_spectra *= ratio
            
            #plot model
            plt.plot(wl, model_spectra, marker='^', label='T_eff='+str(temp*100), alpha=0.6)
        
    
    #plot settings
    plt.figure('spectra'+obs)
    #plt.plot(wl, model_spectra, marker='s', label='stellar model')
    plt.legend()
    plt.xlabel('$\lambda$ $[\mu m]$')
    plt.title(obs)
    
    # plt.figure('snr'+obs)
    # if n_companions > 1:
    #     plt.legend()
    # plt.xlabel('$\lambda$ $[\mu m]$')
    # plt.ylabel('SNR [-]')
    # plt.ylim(bottom=0)
    # plt.title(obs+' snr')
    
    plt.show()
