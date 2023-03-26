import pickle

import numpy as np
import pandas as pd
import os.path

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import socket
import struct

# Prepare SKLearn Models
MAX_ITER = 100
models = [
    ('LR', LogisticRegression(solver='sag', max_iter=MAX_ITER, verbose=1)),
    ('SVM', LinearSVC(penalty='l2', loss='hinge', max_iter=MAX_ITER)),
    ('CART', DecisionTreeClassifier(splitter='random')),
    ('MLP5', MLPClassifier((5,), solver='adam', max_iter=MAX_ITER, shuffle=True, verbose=True, early_stopping=True)),
    ('MLP10', MLPClassifier((10,), solver='adam', max_iter=MAX_ITER, shuffle=True, verbose=True, early_stopping=True)),
    ('MLP40', MLPClassifier((40,), solver='adam', max_iter=MAX_ITER, shuffle=True, verbose=True, early_stopping=True)),
    ('MLP100', MLPClassifier((100,), solver='adam', max_iter=MAX_ITER, shuffle=True, verbose=True, early_stopping=True))
]

# Cross validation method
kfold = model_selection.KFold(n_splits=5, shuffle=True)


# A function to evaluate performance for each test
def evaluate(_classification, _compression, _features, _labels):
    if _classification == 'binary':
        scoring = ('accuracy', 'precision', 'recall', 'f1')
    else:
        scoring = (
            'accuracy', 'precision_micro', 'precision_macro', 'precision_weighted', 'recall_micro', 'recall_macro',
            'recall_weighted', 'f1_micro', 'f1_macro', 'f1_weighted'
        )
    for name, model in models:
        print(_classification, ", Compression : ", _compression, ", Algorithm : ", name, sep='')
        if len(metrics[_classification][_compression][name]['accuracy']):
            print("Skip ... data already available")
        else:
            print("Work in progress ... ", end='')
            results = model_selection.cross_validate(model, _features, _labels, cv=kfold, scoring=scoring)
            metrics[_classification][_compression][name]['fit_time'] = results['fit_time']
            metrics[_classification][_compression][name]['prediction_time'] = results['score_time']
            metrics[_classification][_compression][name]['accuracy'] = results['test_accuracy']
            if _classification == 'binary':
                metrics[_classification][_compression][name]['precision'] = results['test_precision']
                metrics[_classification][_compression][name]['recall'] = results['test_recall']
                metrics[_classification][_compression][name]['f1'] = results['test_f1']
            else:
                metrics[_classification][_compression][name]['precision']['micro'] = results['test_precision_micro']
                metrics[_classification][_compression][name]['precision']['macro'] = results['test_precision_macro']
                metrics[_classification][_compression][name]['precision']['weighted'] = results['test_precision_weighted']
                metrics[_classification][_compression][name]['recall']['micro'] = results['test_recall_micro']
                metrics[_classification][_compression][name]['recall']['macro'] = results['test_recall_macro']
                metrics[_classification][_compression][name]['recall']['weighted'] = results['test_recall_weighted']
                metrics[_classification][_compression][name]['f1']['micro'] = results['test_f1_micro']
                metrics[_classification][_compression][name]['f1']['macro'] = results['test_f1_macro']
                metrics[_classification][_compression][name]['f1']['weighted'] = results['test_f1_weighted']
            print("Done")
            writefile = open("data/results.dat", "wb")
            pickle.dump(metrics, writefile)
            writefile.close()
        accuracy = round(metrics[_classification][_compression][name]['accuracy'].mean() * 100, 2)
        fit = round(metrics[_classification][_compression][name]['fit_time'].min() * 1000)
        predict = round(metrics[_classification][_compression][name]['prediction_time'].min() * 1000)
        print("Accuracy = ", accuracy, "%", sep='')
        print("Fit time = ", fit, "ms", sep='')
        print("Predict time = ", predict, "ms", sep='')


# Main program
if __name__ == '__main__':

    # Load the preprocessed data
    if os.path.isfile("./data/preprocessed_data.npy"):
        print("Preprocessed data already available")
        print("Read the preprocessed data ... ", end='')
        np_dataset = np.load("data/preprocessed_data.npy", allow_pickle=True)
        print("Done")

    else:
        # Read the csv file into a dataframe
        print("Loading the dataset ... ", end='')
        dataframe = pd.read_csv('./data/dataset.csv')
        print("Done")

        print("Convert IP adresses and dates to numerical data ... ", end='')

        # Convert IP adresses to 32-bits integers
        dataframe['Src_IP'] = dataframe['Src_IP'].apply(
            lambda ip: struct.unpack("!I", socket.inet_aton(ip))[0]
        )
        dataframe['Dst_IP'] = dataframe['Dst_IP'].apply(
            lambda ip: struct.unpack("!I", socket.inet_aton(ip))[0]
        )

        # Convert Timestamp to 64-bits integers
        dataframe['Timestamp'] = pd.to_datetime(dataframe['Timestamp']).astype(np.int64)

        print("Done")

        # Extract the labeling features from the dataframe
        binary_labels = dataframe.Label
        category_labels = dataframe.Cat
        subcategory_labels = dataframe.Sub_Cat

        # Keep only numerical data
        dataframe = dataframe.select_dtypes(['number'])

        # Normalize the data
        print("Normalize Numerical Data ... ", end='')
        dataframe = (dataframe.max() - dataframe) / (dataframe.max() - dataframe.min())
        print("Done")

        # Remove columns with NaN values
        print("Remove columns with NaN values ... ", end='')
        nullCols = dataframe.columns[dataframe.isnull().all()].tolist()
        dataframe = dataframe.drop(nullCols, axis=1)
        print("Done")

        # Transform class names into numbers
        print("Managing labels ... ", end='')
        binary_y = binary_labels.replace({"Normal": 0, "Anomaly": 1}).to_numpy(copy=True)
        category_y = category_labels.replace({"Normal": 0, "Mirai": 1, "DoS": 2, "Scan": 3,
                                              "MITM ARP Spoofing": 4}).to_numpy(copy=True)
        subcategory_y = subcategory_labels.replace(
            {"Normal": 0, "Mirai-Ackflooding": 1, "Mirai-Hostbruteforceg": 2, "Mirai-UDP Flooding": 3,
             "Mirai-HTTP Flooding": 4, "DoS-Synflooding": 5, "Scan Port OS": 6, "Scan Hostport": 7,
             "MITM ARP Spoofing": 8}).to_numpy(copy=True)
        dataframe["Label"] = binary_y
        dataframe["Cat"] = category_y
        dataframe["Sub_Cat"] = subcategory_y
        print("Done")

        # Store the normalized data in a file
        print("Store the preprocessed data ... ", end='')
        np_dataset = dataframe.to_numpy()
        np.save("./data/preprocessed_data", np_dataset)
        print("Done")

    # Split the dataset into features and labels
    features = np_dataset[:, :-3]
    binary_y = np_dataset[:, -3]
    category_y = np_dataset[:, -2]
    subcategory_y = np_dataset[:, -1]

    # If the pca data is already available then read it
    if os.path.isfile("data/pca.npy"):
        print("PCA data already available")
        print("Read PCA data ... ", end='')
        pca_features = np.load("data/pca.npy", allow_pickle=True)
        print("Done")

    # Dimension reduction using pca
    else:
        print("Applying PCA to preprocessed data ... ", end='')
        pca = PCA()
        pca.fit(features)
        pca_features = pca.transform(features)
        print("Done")
        print("Storing the pca data in a file ... ", end='')
        np.save("data/pca", pca_features)
        print("Done")

    # Dimension reduction using t-sne
    print("Applying t-SNE to preprocessed data ... ", end='')
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne.fit(features)
    tsne_features = tsne.transform(features)
    print("Done")

    # Dimension reduction using LDA
    print("Applying LDA to preprocessed data ... ", end='')
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(features, subcategory_y)
    lda_features = lda.transform(features)
    print("Done")


    # Prepare a dictionary for metrics
    if not os.path.isfile("data/results.dat"):
        print("Prepare results dictionary ...", end='')
        metrics = {'binary': dict(), 'category': dict(), 'subcategory': dict()}
        for classification in metrics:
            metrics[classification] = {'LR': dict(), 'CART': dict(), 'SVM': dict(),
                               'MLP5': dict(), 'MLP10': dict(), 'MLP40': dict(), 'MLP100': dict()}
            for algorithm in metrics[classification]:
                metrics[classification][algorithm] = {'original': dict(),
                                                      'pca4': dict(), 'pca8': dict(), 'pca16': dict()}
                for compression in metrics[classification][algorithm]:
                    if classification == 'binary':
                        metrics[classification][algorithm][compression] =\
                            {'fit_time': [], 'prediction_time': [], 'accuracy': [],
                             'precision': [], 'recall': [], 'f1': []}
                    else:
                        metrics[classification][algorithm][compression] =\
                            {'fit_time': [], 'prediction_time': [], 'accuracy': []}
                        metrics[classification][algorithm][compression]['precision'] =\
                            {'micro': [], 'macro': [], 'weighted': []}
                        metrics[classification][algorithm][compression]['recall'] =\
                            {'micro': [], 'macro': [], 'weighted': []}
                        metrics[classification][algorithm][compression]['f1'] =\
                            {'micro': [], 'macro': [], 'weighted': []}
        print("Done")

    else:
        print("Result data already exist")
        print("Read old results ... ", end='')
        readfile = open("data/results.dat", "rb")
        metrics = pickle.load(readfile)
        readfile.close()
        print("Done")
    exit(0)

    # Cross validation for the different tests
    # binary classification with no dimensionality reduction
    evaluate('binary', 'original', features, binary_y)
    # category classification with no dimensionality reduction
    evaluate('category', 'original', features, category_y)
    # subcategory classification with no dimensionality reduction
    evaluate('subcategory', 'original', features, subcategory_y)
    # binary classification with pca
    evaluate('binary', 'pca4', pca_features, binary_y)
    evaluate('binary', 'pca6', pca_features, binary_y)
    evaluate('binary', 'pca10', pca_features, binary_y)
    # category classification with pca
    evaluate('category', 'pca4', pca_features, category_y)
    evaluate('category', 'pca6', pca_features, category_y)
    evaluate('category', 'pca10', pca_features, category_y)
    # subcategory classification with pca
    evaluate('subcategory', 'pca4', pca_features, subcategory_y)
    evaluate('subcategory', 'pca6', pca_features, subcategory_y)
    evaluate('subcategory', 'pca10', pca_features, subcategory_y)
