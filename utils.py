# Import the required libraries
import os

# ---------------------- Arguments Class ------------------
# This class is used to store the variables that need to be changed
# while experimentation. Instead or hard coding, this class 
# proves to be handy.

class Args():

    def __init__(self):

        # Initialise the data directory path
        self.directory_path = '../Data/Data/'

        # Initialise the csv directory path
        self.csv_directory_path = '../csv_data/'

        # Initialise the visualisation directory path
        self.visualisation_dir = '../visualisations/'

        # Set if visualisation is required
        self.visualisation = True

        # Set if time series visualisation is required
        self.time_series_visualisation = False

        # Set is pca visualisation is required 
        self.pca = False

        # Set is csv needs to be generated
        self.generate_csv = False

        # Set if dataset statistics are needed
        self.get_statistics = False

        # Set if EMA plot is needed
        self.ema = False

        # Set if correlation map is needed
        self.correlation = False

        # Store a list of all the classifiers needed to be experimented
        self.all_classifiers = ["Random Forest", "Decision Tree", "Neural Network", 
                                "K Nearest Neighbours", "Gaussian Naive Bayes", 
                                "Bernoulli Naive Bayes", "Ada Boost"]

        # Set the classifier type if required
        self.classifier_type = None

        # Set if reading and parsing data is requried
        self.read_data = False

        # Initialise the csv path
        self.csv_path = None

        # Initialise the results directory
        self.results_directory = '../results/'