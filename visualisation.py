# Import the required libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ---------------------- Visualisation Class ---------------------
# This class has function for the following visualisations
#   1. Plotting time series data
#   2. PCA scatter plot visualisation
#   3. Exponential Moving Average visualisation
#   4. Correlation Map

class Visualisation():

    def __init__(self, csv_path, csv_name, visualisation_dir):

        # Store the csv path
        self.csv_path = csv_path

        # Store the csv name
        self.csv_name = csv_name

        # Store the visualisation directory
        self.visualisation_dir = visualisation_dir

        # Read the csv file and store in a dataframe
        self.df = pd.read_csv(csv_path)

        # Fix the x axis as time 
        self.x_axis = 'time'

        # Fix the y axis as the values in the channels
        self.y_axis = ['channel1', 'channel2', 'channel3', 'channel4',
                        'channel5', 'channel6', 'channel7', 'channel8']

    # Function to plot the time series data of each of the channels
    def plot_time_series(self):
        
        # Iterate over each channel
        for y_axis in self.y_axis:
            
            plt.figure(figsize=(8, 6))
            plt.plot(self.df[self.x_axis], self.df[y_axis])
            plt.xlabel(self.x_axis)
            plt.ylabel(y_axis)
            plt.title("Time series for " + y_axis)

            # Save the time series plot in the respective directory
            if os.path.isdir(self.visualisation_dir + self.csv_name):
                plt.savefig(self.visualisation_dir + self.csv_name + '/' + y_axis + '_time_series' + '.png')
            else:
                os.makedirs(self.visualisation_dir + self.csv_name)
                plt.savefig(self.visualisation_dir + self.csv_name + '/' + y_axis + '_time_series' + '.png')

            plt.close()

    # Function to obtain PCA scatter plot along with coloured label of points
    def pca_vis(self):

        # Obtain features
        features = np.array(self.df[self.y_axis])

        # Initialise PCA object with number of components as 2
        pca = PCA(n_components = 2)

        # Fit the PCA model and transform the features
        features_transformed = pca.fit_transform(features)

        # Scatter plot
        plt.figure(figsize=(8,6))
        plt.xlim(min(features_transformed[:, 0]), max(features_transformed[:, 0]))
        plt.ylim(min(features_transformed[:, 1]), max(features_transformed[:, 1]))
        plt.scatter(features_transformed[:, 0], features_transformed[:, 1], c = self.df['class'], s = 3, label = self.df['class'])
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("Scatter plot of first two principal components")
        
        # Save the scatter plot
        if os.path.isdir(self.visualisation_dir + self.csv_name):
            plt.savefig(self.visualisation_dir + self.csv_name + '/' + '_pca_visualisation' + '.png')
        else:
            os.makedirs(self.visualisation_dir + self.csv_name)
            plt.savefig(self.visualisation_dir + self.csv_name + '/' + '_pca_visualisation' + '.png')
            
        plt.close()

    # Function to visualise exponential moving average along with time series
    def ema_vis(self):

        plt.figure(figsize = (8, 6))

        # Iterate over each channel
        for col in self.y_axis:

            self.df[col + '_ema'] = self.df[col].ewm(span = 40, adjust = False).mean()
            plt.plot(self.df['time'], self.df[col], label = 'Time Series')
            plt.plot(self.df['time'], self.df[col + '_ema'], label = 'Exponential Moving Average')
            plt.ylabel("Time Series or EMA for " + col)
            plt.xlabel("Time")
            plt.title("Time Series and EMA Plot")
            plt.legend()

            # Save the plot 
            plt.savefig(self.visualisation_dir + self.csv_name + '/' + 'ema_' + col + '.png')
            plt.close()

    # Function to get the correlation map
    def correlation_map(self):

        f = plt.figure(figsize=(10, 8))
        plt.matshow(self.df[self.y_axis].corr(), fignum=f.number)
        plt.xticks(range(self.df[self.y_axis].select_dtypes(['number']).shape[1]), 
                        self.df[self.y_axis].select_dtypes(['number']).columns, fontsize=9)
        plt.yticks(range(self.df[self.y_axis].select_dtypes(['number']).shape[1]), 
                        self.df[self.y_axis].select_dtypes(['number']).columns, fontsize=9)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize = 10)
        plt.title('Correlation Matrix', fontsize = 12)
        
        # Save the correlation map in required directory
        plt.savefig(self.visualisation_dir + self.csv_name + '/' + 'correlation_map.png')
        plt.close()