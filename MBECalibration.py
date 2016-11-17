"""
Performs an arsenic calibration and can also be used to read previous As calibration files
"""

# TODO: integrate this script into my growth recipes

import ntpath
from datetime import datetime
from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator
from scipy.optimize import fsolve

matplotlib.style.use('ggplot')


class Calibration:
    """
    Class for reading a specific arsenic calibration file
    """

    def __init__(self, material, filepath=None, plot=False):
        """
        Initialize the calibration
        :param filepath: Filename of the desired calibration file to use, if none uses the latest calibration
        """
        self.mat = material
        # Set the name of the x-axis data
        if self.mat == 'As':
            self.x_col = 'AsOpening'
        else:
            self.x_col = '{:s}Temp'.format(self.mat)
        self.spline_interp_inv = None
        if filepath is None:
            filepath = self.get_latest_file(directory='CalibrationFiles/')
        self.directory, self.filename = ntpath.split(filepath)
        self.data = self.read_file(filepath)
        self.spline_interp = self.make_interpolator(self.data)
        self.plt_axis = None

        if plot:
            self.plt_axis = self.plot_data(self.data, self.filename)

    def read_file(self, filename):
        """
        Read in the calibration file
        :param filename: the filename to read in
        :return: data from the calibration file
        """

        data = pd.read_csv(filename, delimiter='\t', header=None, comment='#')  # First try with no headers
        if any(isinstance(cell, str) for cell in data.loc[0]):  # If there is a header
            data = pd.read_csv(filename, delimiter='\t', header=0)  # Import it properly

        data.columns = [self.x_col, 'BFM.P', 'MBE.P']  # Rename dataframe columns

        return data

    def get_latest_file(self, directory=None):
        """
        Get latest calibration file
        :param directory: the directory where the calibration files are stored
        :return: full name and location of the latest calibration file
        """
        if directory is None:
            directory = 'CalibrationFiles/'
        files = glob('{:s}/*_{:s}.txt'.format(directory, self.mat))  # Get all the relevant files in the directory

        files = [list(ntpath.split(filename)) for filename in
                 files]  # Split the paths into directory name and file name

        # Initialize search for the latest file
        latestfile = files[0]
        latest = datetime.strptime(latestfile[1], '%Y-%m-%d_%H-%M-%S_{:s}.txt'.format(self.mat))
        # Loop over all files, extract the latest file
        for pathname, filename in files:
            curr = datetime.strptime(filename, '%Y-%m-%d_%H-%M-%S_{:s}.txt'.format(self.mat))
            if curr > latest:
                latestfile = [pathname, filename]
                latest = datetime.strptime(latestfile[1], '%Y-%m-%d_%H-%M-%S_{:s}.txt'.format(self.mat))

        return latestfile[0] + '/' + latestfile[1]

    def make_interpolator(self, data):
        """
        Makes the interpolator that is used to then interpolate the data
        :param data: the data to be interpolated
        :return: spline interpolation function
        """
        # Average the data points so there is only one y value per x value
        spl_data = data.groupby(by=self.x_col).mean().reset_index()
        spl_x = np.array(spl_data[self.x_col])
        spl_y = np.array(spl_data['BFM.P'])
        # Sort in increasing x values
        order = np.argsort(spl_x)
        spl_x = spl_x[order]
        spl_y = spl_y[order]

        # Create interpolation function
        spl = Akima1DInterpolator(spl_x, spl_y)

        return spl

    def calc_setpoint(self, desired_flux):
        """
        Given the desired material flux, will output the proper setpoint from the latest calibration file
        :param desired_flux: Flux (pressure) that you want to achieve
        :return: Ideal setpoint in the range of the calibration file (0-100 for As, temperature for other cells)
        """

        # TODO: Add the possibility to extrapolate outside fitting range?
        if desired_flux >= self.data['BFM.P'].max():
            raise ValueError('The desired flux is outside of the calibration range!')
        elif desired_flux <= self.data['BFM.P'].min():
            raise ValueError('The desired flux is outside of the calibration range!')

        # Need to "invert" the function: Given a y-value, what is the x-value that we need?
        def spl_inv(x):
            return self.spline_interp(x) - desired_flux

        start_pt = self.data[calib.x_col].mean()  # Place to begin search
        opening = fsolve(spl_inv, start_pt)
        self.spline_interp_inv = spl_inv
        return opening

    def plot_data(self, data, title):
        """
        Plots the calibration data
        :param title: title of the plot
        :param data: data to be plotted
        :return: axis handle
        """
        # Plot the data
        xlim = None
        ylim = None
        # ylim = [-1E-7, max(data['BFM.P']) * 1.05]
        xrange = max(self.data[self.x_col]) - min(self.data[self.x_col])
        yrange = max(self.data['BFM.P']) - min(self.data['BFM.P'])
        ylim = [min(self.data['BFM.P'])-yrange*0.05, max(self.data['BFM.P'])+yrange*0.05]
        xlim = [min(self.data[self.x_col])-xrange*0.05, max(self.data[self.x_col])+xrange*0.05]

        ax = data.plot(x=self.x_col, y='BFM.P', style='.-', grid=True, logy=False, xlim=xlim, ylim=ylim)
        if self.mat == 'As':
            ax.set_xlabel('As Opening %')
        else:
            ax.set_xlabel('{:s} Temp (deg C)'.format(self.mat))
        ax.set_ylabel('Pressure')
        ax.set_title(title)

        # Plot the interpolated function
        x = np.linspace(min(self.data[self.x_col]), max(self.data[self.x_col]), 101)
        y = self.spline_interp(x)
        ax.plot(x, y, label='Interpolation')
        ax.legend(loc='best')

        plt.show()
        return ax


if __name__ == '__main__':
    fn = None
    # fn = 'CalibrationFiles/2016-08-17_01-24-16_As.txt'
    # fn = 'CalibrationFiles/2016-08-16_22-59-32_Ga.txt'
    calib = Calibration('In', filepath=fn, plot=True)
