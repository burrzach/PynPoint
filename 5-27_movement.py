# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 13:00:37 2024

@author: Zach
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

plt.rcParams.update({'font.size': 15})


comp1_sep = 6.072268873502598696e-01
comp1_err = 3.639251782618119105e-04

comp2_sep = 6.441206852782446202e-01
comp2_err = 7.723399380709302124e-04

avg_sep = np.mean([comp1_sep, comp2_sep])
avg_err = np.max([comp1_err, comp2_err])

separations = np.array([[0.9], [0.8], [avg_sep]])
year = np.array([[1991], [2015], [2023]])
error = np.array([0.1, 0.1, avg_err])


regr = LinearRegression()
regr.fit(year, separations, 1/error**2)

plt.errorbar(year.reshape((3)), separations.reshape((3)), yerr=error, color='orange', marker='o', label='observations')
plt.plot(year, regr.predict(year), color='orange', ls='--', label='regression')

plt.xlabel('Year')
plt.ylabel('Separation [arcsec]')
plt.legend()
