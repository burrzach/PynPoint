import pandas as pd

#path to files
sco_cen_file = "D:/Zach/Documents/TUDelft/MSc/Thesis/ScoCenMembers/table7.dat"
phot_file = "D:/Zach/Documents/TUDelft/MSc/Thesis/ScoCenMembers/table1.dat"
target_file = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/observations.csv"
out_file = "D:/Zach/Documents/TUDelft/MSc/Thesis/YSES_IFU/2nd_epoch/star_data.csv"

#column names
table7 = ['num', '2MASS', 'Group', 'Type', 'Ref_type', 'EW(Ha)', 'Ref_EW', 'A_v', 
          'A_v_err', 'pi_kin', 'pi_kin_err', 'log_T', 'log_T_err', 'log_L', 
          'log_L_err', 'age', 'mass', 'disc', 'names']
table1 = ['ID', '2MASS', 'mu_RA', 'mu_RA_err', 'mu_Dec', 'mu_Dec_err', 'Ref_mu', 
          'V', 'V_err', 'Ref_V', 'B-V', 'B-V_err', 'Ref_BV', 'J', 'J_err', 'H', 
          'H_err', 'Ks', 'Ks_err', 'note']

#load data
targets = pd.read_csv(target_file, sep=',', names=['obs','2MASS'])
star_data = pd.read_fwf(sco_cen_file, header=None, names=table7)
phot = pd.read_fwf(phot_file, header=None, names=table1, infer_nrows=20)

#combine dataframes
df = targets.merge(star_data, how='left', on='2MASS')
df = df.merge(phot, how='left', on='2MASS')

#compute distance from paralax
dists = 1 / (df['pi_kin'] * 1e-3)
df.insert(df.columns.get_loc('pi_kin_err')+1, 'dist', dists)

#compute error on distance
err_high = list(abs(df['dist'] - (1e-3 * (df['pi_kin'] + df['pi_kin_err']))**-1))
err_low  = list(abs(df['dist'] - (1e-3 * (df['pi_kin'] - df['pi_kin_err']))**-1))
dists_err = [max(err_high[i], err_low[i]) for i in range(len(err_high))]
df.insert(df.columns.get_loc('dist')+1, 'dist_err', dists_err)


#output data
df.to_csv(out_file, columns=['obs', '2MASS', 'dist', 'dist_err', 'age', 'J', 'J_err'])
