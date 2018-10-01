"""
    This script aims at detecting the fraud transactions from mobile money transfer 
    based on the dataset available at https://www.kaggle.com/ntnu-testimon/paysim1    
"""

import random
from collections import Counter
import numpy as np
from numpy import genfromtxt
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

class ReadContent(object):
    """"
        This is the main class
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def load_content(self):
        """ Below scripts are to get content from all files in each folders """

        my_data = genfromtxt('dataset/kaggle_data_removed_new.csv', delimiter=',')
        my_data = np.asarray(my_data)
        return my_data
    
    def training_SVM(self, training_data, training_label):
        """" Training SVM for classification """
        clf_svm = svm.SVC(kernel='rbf', C=50000, gamma=0.9)
        clf_svm.fit(training_data, training_label)

        return clf_svm
    
    def training_MLP(self, training_data, training_label):
        """" Training Multi Layer Perceptron for classification """
        clf_nn = MLPClassifier(activation= 'relu', solver='lbfgs', alpha=0.1, momentum=0.5, hidden_layer_sizes=(250,250))
        clf_nn.fit(training_data, training_label)

        return clf_nn

    def training_NB(self, training_data, training_label):
        """" Training Naive Bayest for classification """
        clf_gnb = GaussianNB()
        clf_gnb.fit(training_data, training_label)

        return clf_gnb

    def predicting_model(self, model, testing_data):
        """ Testing model """
        predicted_labels = model.predict(testing_data)

        return predicted_labels
        

    def data_division(self, data, num_records, num_each_class_4_test):
        """This method aims at opening file for division"""

        # Shuffle records
        start_range = 1142
        end_range = len(data)
        #print(end_range)
        num_selected = num_records

        random_fraud_idx = random.sample(range(0, 1141), 1141) 
        random_non_fraud_idx = random.sample(range(start_range, end_range), num_selected)


        new_idx = random_fraud_idx+random_non_fraud_idx
        features = np.array(data[new_idx, 0:9])
        labels = np.array(data[new_idx, 10])
       
        unique_label = list(set(labels))
        unique_label = sorted(unique_label, reverse=True)

        label_data = Counter(labels)

        temp0 = random.sample(range(0, label_data[1]), num_each_class_4_test)
        temp1 = random.sample(range(label_data[1], label_data[0]+label_data[1]), num_each_class_4_test)

        all_testing = temp0 + temp1

        testing_data = features[all_testing]
        testing_labels = labels[all_testing]

        training_data = np.delete(features, all_testing, 0)
        training_labels = np.delete(labels, all_testing)


        return training_data, training_labels, testing_data, testing_labels


if __name__ == "__main__":

    CONFIG = {
        'root': 'D:/Fraud_detection',
    }

    # Number of records from each class to be used as testing data
    NUM_TEST_RECORDS = 500

    # Number of total records to use, the original data contains about 1 million records
    NUM_TOTAL_RECORDS = 100000

    TEST = ReadContent(**CONFIG)
    RECORDS = TEST.load_content()

    # Number of independent simulations
    NUM_RUM = 10

    # Allocating memory for some variables 
    TOTAL_SCORE_SVM = []
    TOTAL_SCORE_MLP = []
    TOTAL_SCORE_NB = []

    PRE_NB =[]
    REC_NB = []
    F_SCORE_NB = []

    PRE_MLP = []
    REC_MLP = []
    F_SCORE_MLP = []

    PRE_SVM = []
    REC_SVM = []
    F_SCORE_SVM = []

    for i in range(NUM_RUM):
        print("\n=============", i+1, "=============\n")

        TRAINING_DATA, TRAINING_LABELS, TESTING_DATA, TESTING_LABELS = \
        TEST.data_division(RECORDS, NUM_TOTAL_RECORDS, NUM_TEST_RECORDS)


        scaler = StandardScaler()
        #print(scaler.fit(data))
        scaler.fit(TRAINING_DATA)
        TRAINING_DATA = scaler.transform(TRAINING_DATA)
        TESTING_DATA = scaler.transform(TESTING_DATA)


        MODEL = TEST.training_SVM(TRAINING_DATA, TRAINING_LABELS)
        PREDICTED_LABELS = TEST.predicting_model(MODEL, TESTING_DATA)
        SCORE_SVM = accuracy_score(TESTING_LABELS, PREDICTED_LABELS)
        score_svm = precision_recall_fscore_support(TESTING_LABELS, PREDICTED_LABELS, average='binary')
        PRE_SVM.append(score_svm[0])
        REC_SVM.append(score_svm[1])
        F_SCORE_SVM.append(score_svm[2])
        print('SVM accuracy: ', SCORE_SVM*100)
        TOTAL_SCORE_SVM.append(SCORE_SVM)

        MODEL_MLP = TEST.training_MLP(TRAINING_DATA, TRAINING_LABELS)
        PREDICTED_LABELS = TEST.predicting_model(MODEL_MLP, TESTING_DATA)
        SCORE_MLP = accuracy_score(TESTING_LABELS, PREDICTED_LABELS)
        TOTAL_SCORE_MLP.append(SCORE_MLP)
        score_mlp = precision_recall_fscore_support(TESTING_LABELS, PREDICTED_LABELS, average='binary')
        PRE_MLP.append(score_mlp[0])
        REC_MLP.append(score_mlp[1])
        F_SCORE_MLP.append(score_mlp[2])
        print('MLP accuracy: ', SCORE_MLP*100)
    
        MODEL_NB = TEST.training_NB(TRAINING_DATA, TRAINING_LABELS)
        PREDICTED_LABELS = TEST.predicting_model(MODEL_NB, TESTING_DATA)
        SCORE_NB = accuracy_score(TESTING_LABELS, PREDICTED_LABELS)
        TOTAL_SCORE_NB.append(SCORE_NB)
        score_nb = precision_recall_fscore_support(TESTING_LABELS, PREDICTED_LABELS, average='binary')
        PRE_NB.append(score_nb[0])
        REC_NB.append(score_nb[1])
        F_SCORE_NB.append(score_nb[2])
        print('NB accuracy: ', SCORE_NB*100)
    
    
    print("===========================================\n")
    print('Accuracy\n')
    print(np.average(TOTAL_SCORE_SVM))
    print(np.std(TOTAL_SCORE_SVM))

    print(np.average(TOTAL_SCORE_MLP))
    print(np.std(TOTAL_SCORE_MLP))

    print(np.average(TOTAL_SCORE_NB))
    print(np.std(TOTAL_SCORE_NB))
    print('\n')

    print('Precision \n')
    print(np.average(PRE_SVM))
    print(np.std(PRE_SVM))

    print(np.average(PRE_MLP))
    print(np.std(PRE_MLP))

    print(np.average(PRE_NB))
    print(np.std(PRE_NB))
    print('\n')

    print('Recall \n')
    print(np.average(REC_SVM))
    print(np.std(REC_SVM))

    print(np.average(REC_MLP))
    print(np.std(REC_MLP))

    print(np.average(REC_NB))
    print(np.std(REC_NB))
    print('\n')

    print('F Score \n')
    print(np.average(F_SCORE_SVM))
    print(np.std(F_SCORE_SVM))

    print(np.average(F_SCORE_MLP))
    print(np.std(F_SCORE_MLP))

    print(np.average(F_SCORE_NB))
    print(np.std(F_SCORE_NB))

    LABELS_1 = (TESTING_LABELS > 0).sum()
    print('Numer of frauds in total testing records: ', LABELS_1)
    LABELS_0 = (TESTING_LABELS == 0).sum()
    print('Numer of usual transactions in total testing records: ', LABELS_0)
    print("===========================================\n")
