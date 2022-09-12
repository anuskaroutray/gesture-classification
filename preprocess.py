# Import the required libraries
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import Args
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ----------------- Data reader class -----------------
# This class has member functions that parses the text data 
# and converts it to csv format
# We can also get the dataset statistics from this class 

class DataReader():

    def __init__(self, directory_path, csv_directory_path, generate_csv = False, get_statistics = False):

        # Store the directory path
        self.directory_path = directory_path

        # Store the csv directory path
        self.csv_directory_path = csv_directory_path

        # Store the names of all the directories
        self.directory_contents = os.listdir(self.directory_path)

        # Empty list to store file names
        self.file_names = []

        # Boolean variable to know whether to generate csv or not
        self.generate_csv = generate_csv

        # Boolena variable to know whether to get statistics or not
        self.get_statistics = get_statistics

        if self.generate_csv:
            self.generate_file_paths()
            self.parse_files()
            self.generate_big_csv()

        if self.get_statistics:
            self.get_dataset_statistics()

    # Function to generate the file paths
    def generate_file_paths(self):

        for data_folder in self.directory_contents:
            files = os.listdir(self.directory_path + data_folder)
            self.file_names.append(files)

    # Function to parse all the files
    def parse_files(self):
        
        print("\n--------------------- Parsing Files -------------------------\n")

        # Iterate over the directory containing the text files of each subject
        for file_name, directory_name in tqdm(zip(self.file_names, self.directory_contents), total = len(self.directory_contents)):
            
            # The directory contains two files concerned with two series, so loop over them
            for file in file_name:  
                
                # Open the file
                with open(self.directory_path + directory_name + '/' + file) as f:

                    # Parse the text data    
                    lines = f.readlines()
                    data_dict = {}

                    keys = lines[0].split('\t')
                    keys[-1] = keys[-1][:-1] 
                    for key in keys:
                        data_dict[key] = []

                    for line in lines[1:]:
                        feature_vectors = line.split('\t')
                        for feature_value, key in zip(feature_vectors, keys):
                            if key != 'class':
                                data_dict[key].append(float(feature_value))
                            else:
                                data_dict[key].append(int(feature_value))
                    
                    if len(data_dict['class']) != len(data_dict['channel1']):
                        data_dict['class'].append(0)

                    # Store the parse data in a pandas dataframe
                    df = pd.DataFrame(data_dict)

                    # Save the dataframe in csv format
                    df.to_csv(self.csv_directory_path + file[:-4] + '.csv')

                    # Close the file
                    f.close()

    # Function to concatenate all the csv to generate a large csv
    def generate_big_csv(self):

        print('\n--------------- Combining CSV Files ------------------\n')

        combined_df = pd.concat([pd.read_csv(self.csv_directory_path + csv_file[:-4] + '.csv') 
                                        for csv_file in os.listdir(self.csv_directory_path)])
        
        combined_df.reset_index(inplace = True)
        combined_df.to_csv(self.csv_directory_path + 'combined_data.csv', index = False)

    # Function to obtain the statistics of the datasets and store in csv format for future use
    def get_dataset_statistics(self):

        for csv_file in tqdm(os.listdir(self.csv_directory_path)):
            if "statistics" in csv_file:
                continue
            df = pd.read_csv(self.csv_directory_path + csv_file)
            df_stats = df.describe()
            df_stats.to_csv(self.csv_directory_path + csv_file[:-4] + '_statistics.csv')

# -------------------------- Data Loader Class -----------------------
# This class has the functions that returns the data in the required format to 
# build the machine learning models for classification

class DataLoader():

    def __init__(self, csv_path, ml_classifier = True, scale = True):
        
        # Store the csv path
        self.csv_path = csv_path

        # Boolean variable to know whether data is being prepared to 
        # a machine leanring model
        self.ml_classifier = ml_classifier

        # Read the csv file and store in a dataframe
        self.df = pd.read_csv(self.csv_path)

        # List to store the channel names
        self.channels = ['channel1', 'channel2', 'channel3', 'channel4',
                        'channel5', 'channel6', 'channel7', 'channel8']

        # Boolean variable to check whether that data needs to be scaled or not
        self.scale = scale

    # Function to split the data into train and test set
    def train_test_split_(self, features, labels):
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, random_state = 42)
        return x_train, x_test, y_train, y_test

    # Function to prepare the data in the required format for the ML Model
    def data(self):

        # If ML classifier is true, split and scale data 
        if self.ml_classifier:
            features = np.array(self.df[self.channels])
            labels = np.array(self.df['class'])
            x_train, x_test, y_train, y_test = self.train_test_split_(features, labels)

            if self.scale: 
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.fit_transform(x_test)
            
            return x_train, x_test, y_train, y_test

        # Otherwise, simply return the features and labels
        else:
            features = np.array(self.df[self.channels])
            labels = np.array(self.df['class'])
            return features, labels