# Import the required libraries
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Function to provide reference to class given the class name as string
def str_to_class(str):
    return getattr(sys.modules[__name__], str)

# ---------------------- ML Classifier Class -----------------------
# This class is used to build the machine learning model by fitting 
# on the training data. This is also used to obtain predicted 
# labels on the training as well as the test data.

class MLClassifier():
    
    def __init__(self, classifier_type = 'Random Forest'):

        # Set the classifier type
        self.classifier_type = classifier_type
        
        # Dictionary of all the classifiers along with their class names
        self.classifiers = {"Random Forest": "RandomForestClassifier",
                            "Decision Tree": "DecisionTreeClassifier",
                            "Neural Network": "MLPClassifier",
                            "K Nearest Neighbours": "KNeighborsClassifier",
                            "Gaussian Naive Bayes": "GaussianNB",
                            "Bernoulli Naive Bayes": "BernoulliNB",
                            "Ada Boost": "AdaBoostClassifier"}

        # Create the classifier object with default parameters
        self.cls = str_to_class(self.classifiers[classifier_type])()

    # Function to fit the classifier
    def fit_model(self, x_train, y_train):
        self.cls = self.cls.fit(x_train, y_train)

    # Function to predict on unknown data using the fitted classifier
    def predict(self, x):
        return self.cls.predict(x)