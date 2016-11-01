"""
Performs an arsenic calibration and can also be used to read previous As calibration files
"""

#TODO: create the script that will generate calibration files from the MBE
#TODO: integrate this script into my growth recipes
#TODO: create other scripts for the other elements that should be calibrated

import ntpath
from datetime import datetime
from glob import glob
from scipy.interpolate import Akima1DInterpolator
from scipy.optimize import fsolve
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

matplotlib.style.use('ggplot')


class AsCalibration():
    """
    Class for reading a specific arsenic calibration file
    """

    def __init__(self, filepath=None):
        """
        Initialize the calibration
        :param filename: Filename of the desired calibration file to use, if none uses the latest calibration
        """
        if filepath is None:
            filepath = self.get_latest_file(directory='CalibrationFiles/')
        self.directory, self.filename = ntpath.split(filepath)
        self.data = self.read_file(filepath)
        self.spline_interp = self.make_interpolator(self.data)

        self.pltaxis = self.plot_data(self.data, self.filename)
        self.fit_data(self.data)

    def fit_data(self,data):
        """
        Fits the calibration data with a spline for future interpolation
        :param data:
        :return:
        """

    def read_file(self, filename):
        """
        Read in the calibration file
        :param filename: the filename to read in
        :return: data from the calibration file
        """
        data = pd.read_csv(filename, delimiter='\t', header=None, names=['AsOpening', 'BFM.P', 'MBE.P'])
        return data

    def get_latest_file(self, directory):
        """
        Get latest calibration file
        :param directory: the directory where the calibration files are stored
        :return: full name and location of the latest calibration file
        """
        directory = 'C:/Users/Martin Friedl/Documents/LMSC/Programming/MBECalibrations/CalibrationFiles/'
        files = glob(directory + '/*_As.txt')  # Get all the relevant files in the directory

        files = [list(ntpath.split(filename)) for filename in
                 files]  # Split the paths into directory name and file name

        # Initialize search for the latest file
        latestfile = files[0]
        latest = datetime.strptime(latestfile[1], '%Y-%m-%d_%H-%M-%S_As.txt')
        # Loop over all files, extract the latest file
        for pathname, filename in files:
            curr = datetime.strptime(filename, '%Y-%m-%d_%H-%M-%S_As.txt')
            if curr > latest:
                latestfile = [pathname, filename]
                latest = datetime.strptime(latestfile[1], '%Y-%m-%d_%H-%M-%S_As.txt')

        return latestfile[0] + '/' + latestfile[1]

    def make_interpolator(self, data):
        """
        Makes the interpolator that is used to then interpolate the data
        :param data: the data to be interpolated
        :return: spline interpolation function
        """
        #Average the data points so there is only one y value per x value
        spl_data = data.groupby(by='AsOpening').mean().reset_index()
        spl_x = np.array(spl_data['AsOpening'])
        spl_y = np.array(spl_data['BFM.P'])
        #Sort in increasing x values
        order = np.argsort(spl_x)
        spl_x = spl_x[order]
        spl_y = spl_y[order]

        # Create interpolation function
        spl = Akima1DInterpolator(spl_x,spl_y)

        return spl

    def calc_opening(self, desired_flux):
        """
        Given the desired flux of As, will output the proper opening percentage from the latest calibration file
        :param desired_flux: Flux that you want to achieve
        :return: Proper opening amount of the As valve (0-100)
        """
        #Need to "invert" the function: Given a y-value, what is the x-value that we need?
        spl_inv = lambda x: self.spline_interp(x) - desired_flux
        opening = fsolve(spl_inv,50)
        self.spline_interp_inv = spl_inv

        return opening

    def plot_data(self, data, title):
        """
        Plots the calibration data
        :param data: data to be plotted
        :title title: title of the plot
        :return: axis handle
        """
        #Plot the data
        ylim = [-1E-7,max(data['BFM.P'])*1.05]
        ax = data.plot(x='AsOpening', y='BFM.P', style='.-', grid=True,logy=False,xlim=[-5,105],ylim=ylim)
        ax.set_xlabel('As Opening %')
        ax.set_ylabel('Pressure')
        ax.set_title(title)

        #Plot the interpolated function
        x = np.linspace(0,100,101)
        y = self.spline_interp(x)
        ax.plot(x,y,label='Interpolation')
        ax.legend(loc='best')

        plt.show()
        return ax


if __name__ == '__main__':
    fn = 'CalibrationFiles/2016-08-17_01-24-16_As.txt'
    calib = AsCalibration(fn)
