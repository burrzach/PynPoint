## Imports ##
import numpy as np
import matplotlib.pyplot as plt


## Settings ##
fit_model = True

obs_list = ["2023-05-27", 
            "2023-05-30-2", 
            "2023-06-15-1",
            "2023-07-26-1",
            "2023-08-07-2"
            ]
companion_list = {"2023-05-27":  2, 
                  "2023-05-30-2":1, 
                  "2023-06-15-1":1,
                  "2023-07-26-1":1,
                  "2023-08-07-2":2}
models = {#'BT_5000_45_0_0.txt': '5000', 
          'BT_4500_45_0_0.txt': '4500',
          'BT_4000_45_0_0.txt': '4000',
          'BT_3500_45_0_0.txt': '3500',
          'BT_3000_45_0_0.txt': '3000',
          'BT_2500_45_0_0.txt': '2500',
          'BT_2000_45_0_0.txt': '2000',
          'BT_1500_45_0_0.txt': '1500',}

host_model = 'BT_4500_45_0_0.txt'


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
    comp_file = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/companions/"+obs+"_companion1_data.txt"
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
    #star_spectra = data[1:,1] - data[1:,2]
    #star_error = data[1:,3]
    star_spectra = data[1:,1]
    star_error = np.zeros(star_spectra.shape)
    
    #load stellar model
    star_file = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/companions/star_models/"+host_model
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
    plt.errorbar(wl, star_spectra, yerr=star_error, marker='*', label='host star')
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
        #plt.figure('snr'+obs)
        #plt.plot(wl, snr, marker='o', label='companion 1')
    else:
        plt.errorbar(wl, comp, yerr=error, marker='o', label='companion')
        #plt.figure('snr'+obs)
        #plt.plot(wl, snr, marker='o', label='companion')
        
    
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
        #plt.figure('snr'+obs)
        #plt.plot(wl, snr, marker='o', label='companion '+str(i))
        
    
    #plot each model
    plt.figure('spectra'+obs)
    for model in models.keys():
        #load stellar model
        star_file = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/companions/star_models/"+model
        star_model = np.genfromtxt(star_file)
        
        #bin stellar model
        star_wl = star_model[:,0] / 1e4 #convert angstrom -> micron
        model_spectra = bin_spectra(star_wl, star_model[:,1], wl)
        model_spectra *= ratio
        
        #plot model
        plt.plot(wl, model_spectra, marker='^', label='T_eff='+models[model], alpha=0.6)
        
    
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
