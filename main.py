import pickle
import time

import numpy as np
import pandas as pd
import os
import socket
import struct

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from umap import UMAP

# Prepare SKLearn Models
MAX_ITER = 100
models = [
    ('LR', LogisticRegression(solver='sag', max_iter=MAX_ITER)),
    ('SVM', LinearSVC(penalty='l2', loss='hinge', max_iter=MAX_ITER)),
    ('CART', DecisionTreeClassifier(splitter='random')),
    ('MLP5', MLPClassifier((5,), solver='adam', max_iter=MAX_ITER, shuffle=True, early_stopping=True)),
    ('MLP10', MLPClassifier((10,), solver='adam', max_iter=MAX_ITER, shuffle=True, early_stopping=True)),
    ('MLP40', MLPClassifier((40,), solver='adam', max_iter=MAX_ITER, shuffle=True, early_stopping=True)),
    ('MLP100', MLPClassifier((100,), solver='adam', max_iter=MAX_ITER, shuffle=True, early_stopping=True))
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
        if len(metrics[_classification][name][_compression]['accuracy']):
            print("Skip ... data already available")
        else:
            print("Work in progress ... ", end='')
            results = model_selection.cross_validate(model, _features, _labels, cv=kfold, scoring=scoring)
            metrics[_classification][name][_compression]['fit_time'] = results['fit_time']
            metrics[_classification][name][_compression]['prediction_time'] = results['score_time']
            metrics[_classification][name][_compression]['accuracy'] = results['test_accuracy']
            if _classification == 'binary':
                metrics[_classification][name][_compression]['precision'] = results['test_precision']
                metrics[_classification][name][_compression]['recall'] = results['test_recall']
                metrics[_classification][name][_compression]['f1'] = results['test_f1']
            else:
                metrics[_classification][name][_compression]['precision']['micro'] = results['test_precision_micro']
                metrics[_classification][name][_compression]['precision']['macro'] = results['test_precision_macro']
                metrics[_classification][name][_compression]['precision']['weighted'] = results['test_precision_weighted']
                metrics[_classification][name][_compression]['recall']['micro'] = results['test_recall_micro']
                metrics[_classification][name][_compression]['recall']['macro'] = results['test_recall_macro']
                metrics[_classification][name][_compression]['recall']['weighted'] = results['test_recall_weighted']
                metrics[_classification][name][_compression]['f1']['micro'] = results['test_f1_micro']
                metrics[_classification][name][_compression]['f1']['macro'] = results['test_f1_macro']
                metrics[_classification][name][_compression]['f1']['weighted'] = results['test_f1_weighted']
            print("Done")
            writefile = open("data/results.dat", "wb")
            pickle.dump(metrics, writefile)
            writefile.close()
        accuracy = round(metrics[_classification][name][_compression]['accuracy'].mean() * 100, 2)
        fit = round(metrics[_classification][name][_compression]['fit_time'].min() * 1000)
        predict = round(metrics[_classification][name][_compression]['prediction_time'].min() * 1000)
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

        print("Convert IP addresses and dates to numerical data ... ", end='')

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

    # If the UMAP data for 3 dimensions is already available then read it
    if os.path.isfile("data/umap3.npy"):
        print("UMAP data for 3 dimensions already available")
        print("Read UMAP data for 3 dimensions ... ", end='')
        umap3_features = np.load("data/umap3.npy", allow_pickle=True)
        print("Done")

    else:
        # Dimension reduction to 3 dimensions using UMAP
        print("Applying UMAP (3 dimensions) to preprocessed data ... ", end='')
        # Random state argument is used for reproducibility
        ump = UMAP(n_components=3, n_jobs=4, init="random", random_state=42, min_dist=0)
        ump.fit(features)
        umap3_features = ump.transform(features)
        print("Done")
        print("Storing the UMAP data (3 dimensions) in a file ... ", end='')
        np.save("data/umap3", umap3_features)
        print("Done")

    # If the UMAP data for 6 dimensions is already available then read it
    if os.path.isfile("data/umap6.npy"):
        print("UMAP data for 6 dimensions already available")
        print("Read UMAP data for 6 dimensions ... ", end='')
        umap6_features = np.load("data/umap6.npy", allow_pickle=True)
        print("Done")

    else:
        # Dimension reduction to 6 dimensions using UMAP
        print("Applying UMAP (6 dimensions) to preprocessed data ... ", end='')
        ump = UMAP(n_components=6, n_jobs=4, init="random", random_state=42, min_dist=0)
        ump.fit(features)
        umap6_features = ump.transform(features)
        print("Done")
        print("Storing the UMAP data (6 dimensions) in a file ... ", end='')
        np.save("data/umap6", umap6_features)
        print("Done")

    # If the UMAP data for 8 dimensions is already available then read it
    if os.path.isfile("data/umap8.npy"):
        print("UMAP data for 8 dimensions already available")
        print("Read UMAP data for 8 dimensions ... ", end='')
        umap8_features = np.load("data/umap8.npy", allow_pickle=True)
        print("Done")

    else:
        # Dimension reduction to 8 dimensions using UMAP
        print("Applying UMAP (8 dimensions) to preprocessed data ... ", end='')
        ump = UMAP(n_components=8, n_jobs=4, init="random", random_state=42, min_dist=0)
        ump.fit(features)
        umap8_features = ump.transform(features)
        print("Done")
        print("Storing the UMAP data (8 dimensions) in a file ... ", end='')
        np.save("data/umap8", umap8_features)
        print("Done")

    # If the PCA data is already available then read it
    if os.path.isfile("data/pca.npy"):
        print("PCA data already available")
        print("Read PCA data ... ", end='')
        pca_features = np.load("data/pca.npy", allow_pickle=True)
        print("Done")

    # Dimension reduction using PCA
    else:
        print("Applying PCA to preprocessed data ... ", end='')
        pca = PCA()
        pca.fit(features)
        pca_features = pca.transform(features)
        print("Done")
        print("Storing the PCA data in a file ... ", end='')
        np.save("data/pca", pca_features)
        print("Done")

    # If the LDA data is already available then read it
    if os.path.isfile("data/lda.npy"):
        print("LDA data already available")
        print("Read LDA data ... ", end='')
        lda_features = np.load("data/lda.npy", allow_pickle=True)
        print("Done")

    # Dimension reduction using LDA
    else:
        print("Applying LDA to preprocessed data ... ", end='')
        lda = LinearDiscriminantAnalysis()
        lda.fit(features, subcategory_y)
        lda_features = lda.transform(features)
        print("Done")
        print("Storing the LDA data in a file ... ", end='')
        np.save("data/lda", lda_features)
        print("Done")

    # Prepare a dictionary for metrics
    if os.path.isfile("data/results.dat"):
        print("Result data already exist")
        print("Read old results ... ", end='')
        readfile = open("data/results.dat", "rb")
        metrics = pickle.load(readfile)
        readfile.close()
        print("Done")

    else:
        print("Prepare results dictionary ...", end='')
        metrics = {'binary': dict(), 'category': dict(), 'subcategory': dict()}
        for classification in metrics:
            metrics[classification] = {'LR': dict(), 'CART': dict(), 'SVM': dict(),
                                       'MLP5': dict(), 'MLP10': dict(), 'MLP40': dict(), 'MLP100': dict()}
            for algorithm in metrics[classification]:
                metrics[classification][algorithm] = {'original': dict(),
                                                      'pca3': dict(), 'pca6': dict(), 'pca8': dict(),
                                                      'lda3': dict(), 'lda6': dict(), 'lda8': dict(),
                                                      'umap3': dict(), 'umap6': dict(), 'umap8': dict()}
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

    # Cross validation for the different tests
    # binary classification with no dimensionality reduction
    evaluate('binary', 'original', features, binary_y)
    # category classification with no dimensionality reduction
    evaluate('category', 'original', features, category_y)
    # subcategory classification with no dimensionality reduction
    evaluate('subcategory', 'original', features, subcategory_y)
    # binary classification with UMAP
    evaluate('binary', 'umap3', umap3_features, binary_y)
    evaluate('binary', 'umap6', umap6_features, binary_y)
    evaluate('binary', 'umap8', umap8_features, binary_y)
    # category classification with UMAP
    evaluate('category', 'umap3', umap3_features, category_y)
    evaluate('category', 'umap6', umap6_features, category_y)
    evaluate('category', 'umap8', umap8_features, category_y)
    # subcategory classification with UMAP
    evaluate('subcategory', 'umap3', umap3_features, subcategory_y)
    evaluate('subcategory', 'umap6', umap6_features, subcategory_y)
    evaluate('subcategory', 'umap8', umap8_features, subcategory_y)
    # binary classification with PCA
    evaluate('binary', 'pca3', pca_features[:, :3], binary_y)
    evaluate('binary', 'pca6', pca_features[:, :6], binary_y)
    evaluate('binary', 'pca8', pca_features[:, :8], binary_y)
    # category classification with PCA
    evaluate('category', 'pca3', pca_features[:, :3], category_y)
    evaluate('category', 'pca6', pca_features[:, :6], category_y)
    evaluate('category', 'pca8', pca_features[:, :8], category_y)
    # subcategory classification with PCA
    evaluate('subcategory', 'pca3', pca_features[:, :3], subcategory_y)
    evaluate('subcategory', 'pca6', pca_features[:, :6], subcategory_y)
    evaluate('subcategory', 'pca8', pca_features[:, :8], subcategory_y)
    # binary classification with LDA
    evaluate('binary', 'lda3', lda_features[:, :3], binary_y)
    evaluate('binary', 'lda6', lda_features[:, :6], binary_y)
    evaluate('binary', 'lda8', lda_features[:, :8], binary_y)
    # category classification with LDA
    evaluate('category', 'lda3', lda_features[:, :3], category_y)
    evaluate('category', 'lda6', lda_features[:, :6], category_y)
    evaluate('category', 'lda8', lda_features[:, :8], category_y)
    # subcategory classification with LDA
    evaluate('subcategory', 'lda3', lda_features[:, :3], subcategory_y)
    evaluate('subcategory', 'lda6', lda_features[:, :6], subcategory_y)
    evaluate('subcategory', 'lda8', lda_features[:, :8], subcategory_y)
