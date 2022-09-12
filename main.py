# Import the required libraries
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import Args
from model import MLClassifier
from metric import EvaluationMetrics
from visualisation import Visualisation
from preprocess import DataLoader, DataReader

# Function to build the machine learning classifier
def build_ml_classifier(sg):

    # Create an object of ML Classifier
    cls = MLClassifier(classifier_type = sg.classifier_type)

    # Load the data object in the required format
    data_loader = DataLoader(sg.csv_path, ml_classifier = True, scale = True)

    # Obtain the train and test data
    x_train, x_test, y_train, y_test = data_loader.data()

    # Fit the model
    cls.fit_model(x_train, y_train)

    # Obtain predictions on the training and test set
    y_train_preds = cls.predict(x_train)
    y_test_preds = cls.predict(x_test)

    # Create an object for training evaluation metrics
    evaluation_metrics = EvaluationMetrics(y_train = y_train, y_test = y_test, y_train_preds = y_train_preds, 
                                            y_test_preds = y_test_preds, train = True)
    
    # Obtain evaluation metrics for training data
    train_accuracy = evaluation_metrics.accuracy()
    train_precision = evaluation_metrics.precision()
    train_recall = evaluation_metrics.recall()
    train_f1score = evaluation_metrics.f1()

    # Create an object for testing evaluation metrics 
    evaluation_metrics = EvaluationMetrics(y_train = y_train, y_test = y_test, y_train_preds = y_train_preds, 
                                            y_test_preds = y_test_preds, train = False)
    
    # Obtain evaluation metrics for test data
    test_accuracy = evaluation_metrics.accuracy()
    test_precision = evaluation_metrics.precision()
    test_recall = evaluation_metrics.recall()
    test_f1score = evaluation_metrics.f1()

    # Return the evaluation metrics as a tuple
    return (train_accuracy, train_precision, train_recall, train_f1score,
            test_accuracy, test_precision, test_recall, test_f1score)

# Function to invoke visualisations and save them
def save_visualisation(sg):

    # Code for the time series visualisation of all the csv
    if sg.time_series_visualisation:
        for csv_file in tqdm(os.listdir(sg.csv_directory_path)):
            if "statistics" in csv_file or "combined" in csv_file:
                continue
            vis = Visualisation(sg.csv_directory_path + csv_file, csv_file[:-4], sg.visualisation_dir)
            vis.plot_time_series()
    
    # Code for the scatter plot of PCA of all the csv
    if sg.pca:
        for csv_file in tqdm(os.listdir(sg.csv_directory_path)):
            if "statistics" in csv_file or "combined" in csv_file:
                continue
            vis = Visualisation(sg.csv_directory_path + csv_file, csv_file[:-4], sg.visualisation_dir)
            vis.pca_vis()

    # Code for the time series and EMA visualisation of all the csv
    if sg.ema:
        for csv_file in tqdm(os.listdir(sg.csv_directory_path)):
            if "statistics" in csv_file or "combined" in csv_file:
                continue
            vis = Visualisation(sg.csv_directory_path + csv_file, csv_file[:-4], sg.visualisation_dir)
            vis.ema_vis()

    # Code for the correlation map of all the csv
    if sg.correlation:
        for csv_file in tqdm(os.listdir(sg.csv_directory_path)):
            if "statistics" in csv_file or "combined" in csv_file:
                continue
            vis = Visualisation(sg.csv_directory_path + csv_file, csv_file[:-4], sg.visualisation_dir)
            vis.correlation_map()

# Main function invoking all the required functions
def main():

    # Create object of argument class
    sg = Args()

    # For visualisation
    if sg.visualisation:
        save_visualisation(sg)

    # For reading and parsing data
    if sg.read_data:
        data_reader = DataReader(sg.directory_path, sg.csv_directory_path, 
                                generate_csv = sg.generate_csv, get_statistics = sg.get_statistics) 

    # For obtaining data statistics
    if sg.get_statistics:
        data_reader.get_dataset_statistics()
    
    # Fit all the models on all the csv files
    for csv_file in tqdm(os.listdir(sg.csv_directory_path)):

        # Dictionary to store the results
        results_dict = {"Classifier": sg.all_classifiers}

        train_accuracy = []    # List to store training accuracy
        train_precision = []   # List to store the training precision
        train_recall = []      # List to store the training recall
        train_f1score = []     # List to store the training f1 score
        test_accuracy = []     # List to store the test accuracy
        test_precision = []    # List to store the test precision
        test_recall = []       # List to store the test recall
        test_f1score = []      # List to store the test f1 score

        if "statistics" in csv_file or "combined" in csv_file:
            continue

        # Loop over all the classifiers
        for classifier in sg.all_classifiers:
            
            # Define the classifier
            sg.classifier_type = classifier
            sg.csv_path = sg.csv_directory_path + csv_file

            # Obtain the evaluation scores
            score = build_ml_classifier(sg)
            train_accuracy.append(score[0])
            train_precision.append(score[1])
            train_recall.append(score[2])
            train_f1score.append(score[3])
            test_accuracy.append(score[4])
            test_precision.append(score[5])
            test_recall.append(score[6])
            test_f1score.append(score[7])
        
        # Store the evaluation score in the results dictionary
        results_dict["Train Accuracy"] =  train_accuracy
        results_dict["Train Precision"] = train_precision
        results_dict["Train Recall"] = train_recall
        results_dict["Train F1 Score"] = train_f1score
        results_dict["Test Accuracy"] = test_accuracy
        results_dict["Test Precision"] = test_precision
        results_dict["Test Recall"] = test_recall
        results_dict["Test F1 Score"] = test_f1score
        
        # Save the results dictionary as csv file
        results_df = pd.DataFrame(results_dict)
        if not os.path.isdir(sg.results_directory):
            os.makedirs(sg.results_directory)
        results_df.to_csv(sg.results_directory + csv_file[:-4] + '_results.csv')
    
if __name__ == '__main__':
    main()