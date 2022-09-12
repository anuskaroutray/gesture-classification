# Import the required libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------------- Evaluation Metrics Class -----------------------
# This class has the following metrics
#   1. Accuracy Score
#   2. Precision Score
#   3. Recall Score
#   4. F1 Score

class EvaluationMetrics():

    def __init__(self, y_train = None, y_test = None, y_train_preds = None, y_test_preds = None, train = False):

        self.y_train = y_train               # Initialise actual train labels
        self.y_test = y_test                 # Initialise predicted train labels
        self.y_train_preds = y_train_preds   # Initialise actual test labels
        self.y_test_preds = y_test_preds     # Initialise predicted test labels
        self.train = train

    # Function to compute accuracy
    def accuracy(self):
        if self.train:
            return accuracy_score(self.y_train, self.y_train_preds)

        return accuracy_score(self.y_test, self.y_test_preds)

    # Function to compute precision
    def precision(self):
        if self.train:
            return precision_score(self.y_train, self.y_train_preds, average = 'macro')

        return precision_score(self.y_test, self.y_test_preds, average = 'macro')

    # Function to compute recall
    def recall(self):
        if self.train:
            return recall_score(self.y_train, self.y_train_preds, average = 'macro')

        return recall_score(self.y_test, self.y_test_preds, average = 'macro')
    
    # Function to compute f1 score
    def f1(self):
        if self.train:
            return f1_score(self.y_train, self.y_train_preds, average = 'macro')

        return f1_score(self.y_test, self.y_test_preds, average = 'macro')
