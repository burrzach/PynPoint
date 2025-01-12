## Imports ##
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


## Settings ##
fit_model = True      #fit host spectrum to match model
plot_models = True    #plot range of model spectra to compare to companion
plot_host = True      #plot host star spectrum
fit_companion = False  #scale models to value of companion when plotting
find_best_fit = False  #find model which fits closest to companion spectrum
calc_distance = False  #calculate true distance based off best fit temperature
binary_scaling = False #halve brightness (for binary companions)
residuals = False      #use data from measuring after subtraction

temp_range = range(40, 24, -5) #range of temperatures to plot models
#temp_range = range(60, 3, -1) #all models for best fitting
#temp_range = []

#temp_range = [48] + list(temp_range)

plt.rcParams.update({'font.size': 15})

obs_list = [#"2023-05-27",    #which observations to plot
            #"2023-05-30-2", 
            "2023-06-15-1",
            #"2023-07-26-1",
            #"2023-08-07-2"
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
comp_fit_temp = {"2023-05-27":  [42, 42], #best fit model for each companion
                 "2023-05-30-2":[60],
                 "2023-06-15-1":[34],
                 "2023-07-26-1":[56],
                 "2023-08-07-2":[34, 34]}

folder = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/companions/"
props_file = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/star_data.csv"

exclude = None #exclude certain data points when doing the best fit determination
#exclude = range(12,23)

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

def RMSE(spectra, model, exclude=None):
    
    if exclude is not None:
        spectra = np.delete(spectra, exclude)
        model = np.delete(model, exclude)
    
    scaling = np.mean(spectra[1:-1]) / np.mean(model[1:-1])
    
    return np.sqrt(np.mean((spectra[1:-1] - model[1:-1]*scaling)**2)) / np.mean(spectra[1:-1])

def load_model(temp, wl):
    
    if temp < 10:
        temp = '0'+str(temp)
    
    #load stellar model
    star_file = folder+"star_models/BT_0"+str(temp)+"_45_0_0.txt"
    star_model = np.genfromtxt(star_file)
    
    #bin stellar model
    star_wl = star_model[:,0] / 1e4 #convert angstrom -> micron
    model_spectra = bin_spectra(star_wl, star_model[:,1], wl)
    
    return model_spectra

def apparent_to_absolute(mag, distance):
    return mag - 5*np.log10(distance) + 5

def calc_dist(apparent, absolute):
    return 10 ** ((apparent-absolute+5) / 5)


fitting_dict = {}
noise_dict = {}
host_dict = {}

#perform calculations and plotting for all observations
for obs in obs_list:
    n_companions = companion_list[obs]
    
    ## Analysis ##
    #load companion data
    comp_file = folder+obs+"_companion1_" + "resid_"*residuals + "data.txt"
    data = np.genfromtxt(comp_file)
    comp1data = data
    
    
    #print stats
    print('\n'+obs)
    print('sep, angle, snr, fpf, mag')
    if n_companions > 1:
        print('companion 1:')
    head = data[0,[7,9,11,12,13]]
    print(head)
    
    
    #format spectra
    wl = data[1:,0] / 1e3 #convert nm -> micron
    wl.sort()
    star_spectra = data[1:,1] - data[1:,2]
    star_error = data[1:,3]
    
    host_dict[obs] = data[1:,1]
    noise_dict[obs] = data[1:,2]
    
    #load all spectra models for later
    model_dict = {}
    for temp in temp_range:
        model_dict[temp] = load_model(temp, wl)
    if host_temp[obs] not in temp_range:
        model_dict[host_temp[obs]] = load_model(host_temp[obs], wl)
    for temp in comp_fit_temp[obs]:
        if temp not in temp_range:
            model_dict[temp] = load_model(temp, wl)
    
    #load model spectra
    model_spectra = np.copy(model_dict[host_temp[obs]])
    ratio = np.mean(star_spectra[1:-1]) / np.mean(model_spectra[1:-1])
    model_spectra *= ratio
    if fit_model == False:
        model_spectra *= 3
        fitting = 1
    else:
        #fit star spectrum to model, then use to correct companion spectrum
        fitting = model_spectra / star_spectra
        star_spectra *= fitting
        star_error *= fitting
        
    fitting_dict[obs] = fitting
    
    
    #make figure and plot star
    plt.figure('spectra'+obs)
    plt.yscale('log')
    if plot_host:
        plt.errorbar(wl, star_spectra, yerr=star_error, marker='*', 
                     label='host star (T_eff='+str(host_temp[obs]*100)+'K)')
    
    #plot first companion
    comp = data[1:,4] - data[1:,6]
    error = abs(data[1:,5])
    snr = data[1:,11]
    
    # if obs == "2023-06-15-1" or obs == "2023-05-27" or obs == "2023-08-07-2": #!!!
    #     binary_scaling = True
    # else:
    #     binary_scaling = False
    
    if obs == "2023-06-15-1":
        error *= 0.5
    
    if binary_scaling:
        comp *= 0.5 #divide by 2 because companion is a binary
    
    if fit_model == True:
        comp *= fitting
        error *= fitting
    if n_companions > 1:
        #plt.errorbar(wl, comp, yerr=error, marker='s', label='companion 1 (combined brightness)', color='orange', alpha=0.5) #!!!
        plt.errorbar(wl, comp, yerr=error, marker='o', label='companion 1', color='orange')
        plt.figure('snr'+obs)
        plt.plot(wl, snr, marker='o', label='companion 1', color='orange')
    else:
        plt.errorbar(wl, comp, yerr=error, marker='s', label='companion (combined brightness)', color='orange', alpha=0.5) #!!!
        #plt.errorbar(wl, comp, yerr=error, marker='o', label='companion', color='orange')
        plt.figure('snr'+obs)
        plt.plot(wl, snr, marker='o', label='companion', color='orange')
    
    #calculate goodness of fit for each model
    if find_best_fit:
        model_goodness = np.empty((len(temp_range)))
        for i, temp in enumerate(temp_range):
            model_spectra = model_dict[temp]
            
            #calculate goodness of fit
            model_goodness[i] = RMSE(comp, model_spectra, exclude)
        
        #find best fit
        i_best = np.argmin(model_goodness)
        best_fit_temp = temp_range[i_best]
        best_fit_goodness = np.round(model_goodness[i_best], 3)
        
        #reload best fit
        model_spectra = model_dict[best_fit_temp]
        scaling = np.mean(comp[1:-1]) / np.mean(model_spectra[1:-1])
        model_spectra_scaled = model_spectra * scaling
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        if n_companions > 1:
            ax1.errorbar(wl, comp, yerr=error, marker='o', label='companion 1', color='orange')
            label=f'companion 1 best fit\n(T_eff={best_fit_temp*100}, NRMSE={best_fit_goodness})'
            ax1.plot(wl, model_spectra_scaled, label=label, color='red', alpha=0.6)
            
            ax2.plot(np.array(temp_range)*100, model_goodness, label='companion 1', color='orange')
        else:
            ax1.errorbar(wl, comp, yerr=error, marker='o', label='companion', color='orange')
            label=f'companion best fit\n(T_eff={best_fit_temp*100}, NRMSE={best_fit_goodness})'
            ax1.plot(wl, model_spectra_scaled, label=label, color='red', alpha=0.6)
            
            ax2.plot(np.array(temp_range)*100, model_goodness, label='companion', color='orange')
            
    #calculate distance to companion
    if calc_distance:
        star_props = pd.read_csv(props_file, index_col=0)
        star_app_mag = star_props.loc[star_props['obs'] == obs, 'J'].iloc[0]
        dist = star_props.loc[star_props['obs'] == obs, 'dist'].iloc[0]
        
        dmag = data[0,13]
        app_mag_comp = dmag + star_app_mag
        
        best_fit = model_dict[comp_fit_temp[obs][0]] * ratio * fitting
        if binary_scaling:
            model_dmag = -2.5 * np.log10(sum(best_fit[2:-2])*2 /  sum(star_spectra[2:-2]))
        else:
            model_dmag = -2.5 * np.log10(sum(best_fit[2:-2]) /  sum(star_spectra[2:-2]))
        model_abs_mag = apparent_to_absolute(model_dmag + star_app_mag, dist)
        
        comp_dist = calc_dist(app_mag_comp, model_abs_mag)
        print('Host star distance: ', np.round(dist,3), ' pc')
        print('Companion 1 true distance: ', np.round(comp_dist,3), ' pc')
    
    #determine scaling for models if necessary
    if fit_companion:
        #load stellar model
        temp = int(np.round(np.mean(temp_range),0))
        model_spectra = model_dict[temp]
        model_spectra = model_dict[temp_range[0]] #!!!
        
        #calculate ratio
        ratio = np.mean(comp[1:-1]) / np.mean(model_spectra[1:-1])
        
    
    #repeat for each companion beyond the first
    comp1 = comp #save data on 1st companion first to allow comparing
    for i in range(1, n_companions+1): #!!!
        #load companion data
        comp_file = folder + obs + f"_companion{i}_" + "resid_"*residuals + "data.txt"
        data = np.genfromtxt(comp_file)
        
        head = data[0,[7,9,11,12,13]]
        print('companion '+str(i)+':\n', head)
        
        comp = data[1:,4] - data[1:,6]
        error = data[1:,5]
        snr = data[1:,11]
        
        if fit_model == True:
            comp *= fitting
            error *= fitting
        
        if i == 1: #!!!
            comp *= 0.5
            plt.figure('spectra'+obs)
            plt.errorbar(wl, comp, yerr=error, marker='o', label='companion 1 (halved)', color='orange')
        else:
            plt.figure('spectra'+obs)
            plt.errorbar(wl, comp, yerr=error, marker='o', label=f'companion {i}', color='green')
            plt.figure('snr'+obs)
            plt.plot(wl, snr, marker='o', label='companion '+str(i), color='green')
        
        #calculate goodness of fit for each model
        if find_best_fit:
            model_goodness = np.empty((len(temp_range)))
            for j, temp in enumerate(temp_range):
                model_spectra = model_dict[temp]
                
                #calculate goodness of fit
                model_goodness[j] = RMSE(comp, model_spectra, exclude)
            
            #find best fit
            i_best = np.argmin(model_goodness)
            best_fit_temp = temp_range[i_best]
            best_fit_goodness = np.round(model_goodness[i_best], 3)
            
            #reload best fit
            model_spectra = model_dict[best_fit_temp]
            scaling = np.mean(comp[1:-1]) / np.mean(model_spectra[1:-1])
            model_spectra_scaled = model_spectra * scaling
            
            ax1.errorbar(wl, comp, yerr=error, marker='o', label=f'companion {i}', color='green')
            label=f'companion {i} best fit\n(T_eff={best_fit_temp*100}, NRMSE={best_fit_goodness})'
            ax1.plot(wl, model_spectra_scaled, label=label, color='blue', alpha=0.6)
            
            ax2.plot(np.array(temp_range)*100, model_goodness, label=f'companion {i}', color='green')
            
            comp_comparison = RMSE(comp1, comp)
            print(f'Companion {i} vs 1 NRMSE: {comp_comparison}')
            
        #calculate distance to companion
        if calc_distance:
            star_props = pd.read_csv(props_file, index_col=0)
            star_app_mag = star_props.loc[star_props['obs'] == obs, 'J'].iloc[0]
            dist = star_props.loc[star_props['obs'] == obs, 'dist'].iloc[0]
            
            dmag = data[0,13]
            app_mag_comp = dmag + star_app_mag
            
            best_fit = model_dict[comp_fit_temp[obs][i-1]] * ratio * fitting
            model_dmag = -2.5 * np.log10(sum(best_fit[2:-2]) /  sum(star_spectra[2:-2]))
            model_abs_mag = apparent_to_absolute(model_dmag + star_app_mag, dist)
            
            comp_dist = calc_dist(app_mag_comp, model_abs_mag)
            print(f'Companion {i} true distance: ', np.round(comp_dist,3), ' pc')
            
        comp2 = comp #save for analysis after the fact
        
    
    #plot each model
    if plot_models:
        plt.figure('spectra'+obs)
        model_goodness = np.empty((len(temp_range)))
        for temp in temp_range:
            model_spectra = model_dict[temp] * ratio
            if temp == np.nan:
                plt.plot(wl, model_spectra, alpha=0.6, color='red', label='Best fit for companion 1 (T_eff='+str(temp*100)+')')
                plt.text(wl[-2], model_spectra[-2]*0.95, f'{temp*100}K') #!!!
            elif temp == np.inf:
                plt.plot(wl, model_spectra, alpha=0.6, color='blue', label='Best fit for companion 2 (T_eff='+str(temp*100)+')')
                plt.text(wl[-2], model_spectra[-2]*0.95, f'{temp*100}K') #!!!
            else:
                plt.plot(wl, model_spectra, alpha=0.6, color='black') #label='T_eff='+str(temp*100)
                plt.text(wl[-2], model_spectra[-2]*0.95, f'{temp*100}K') #!!!
            
    
    #plot settings
    plt.figure('spectra'+obs)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(Line2D([],[],color='black',alpha=0.6))
    labels.append('Other model spectra')
    plt.legend(handles, labels)
    plt.xlabel('$\lambda$ $[\mu m]$')
    plt.ylabel('Flux [erg/$cm^2$/s/A]')
    plt.title(obs)
    
    plt.figure('snr'+obs)
    if n_companions > 1:
        plt.legend()
    plt.xlabel('$\lambda$ $[\mu m]$')
    plt.ylabel('SNR [-]')
    plt.ylim(bottom=0)
    plt.title(obs+' SNR')
    
    if find_best_fit:
        ax1.set_yscale('log')
        ax1.legend()
        ax1.set_xlabel('$\lambda$ $[\mu m]$')
        ax1.set_ylabel('Flux [erg/$cm^2$/s/A]')
        
        if n_companions > 1:
            ax2.legend()
        ax2.set_xlabel('T_eff [K]')
        ax2.set_ylabel('RMSE [-]')
        
        ax1.set_title(obs)
