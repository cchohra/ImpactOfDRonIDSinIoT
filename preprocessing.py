
# This module is used to preprocess the data and apply dimensionality reduction.
import numpy as np
import pandas as pd
import os
import socket
import struct

from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# A function to preprocess the data.
def preprocess(source_file, target_file):

    # If the preprocessed data is available, read it
    if os.path.isfile(target_file + ".npy"):
        print("Preprocessed data already available")
        print("Read the preprocessed data ... ", end="")
        preprocessed_data = np.load(target_file + ".npy", allow_pickle=True)
        print("Done")

    else:
        # Read the csv file into a dataframe
        print("Loading the dataset ... ", end="")
        dataframe = pd.read_csv(source_file)
        # Keep the first 10000 rows for testing
        dataframe = dataframe.head(10000)
        print("Done")

        print("Convert IP addresses and dates to numerical data ... ", end="")

        # Convert IP addresses to 32-bits integers
        dataframe["Src_IP"] = dataframe["Src_IP"].apply(
            lambda ip: struct.unpack("!I", socket.inet_aton(ip))[0]
        )
        dataframe["Dst_IP"] = dataframe["Dst_IP"].apply(
            lambda ip: struct.unpack("!I", socket.inet_aton(ip))[0]
        )

        # Convert Timestamp to 64-bits integers
        dataframe["Timestamp"] = pd.to_datetime(dataframe["Timestamp"]).astype(np.int64)

        print("Done")

        # Extract the labeling features from the dataframe
        binary_labels = dataframe.Label
        category_labels = dataframe.Cat
        subcategory_labels = dataframe.Sub_Cat

        # Keep only numerical data
        dataframe = dataframe.select_dtypes(["number"])

        # Replace infinities by column max (and min) depending on the sign
        print("Replace infinities by column max ... ", end="")
        dataframe = dataframe.replace(np.inf, np.nan)
        dataframe = dataframe.fillna(dataframe.max())
        dataframe = dataframe.replace(-np.inf, np.nan)
        dataframe = dataframe.fillna(dataframe.min())
        print("Done")

        # Scale the data
        print("Scale the data ... ", end="")
        dataframe = (dataframe.max() - dataframe) / (dataframe.max() - dataframe.min())
        print("Done")

        # Remove columns with all NaN values
        print("Remove columns with NaN values ... ", end="")
        null_cols = dataframe.columns[dataframe.isnull().all()].tolist()
        dataframe = dataframe.drop(null_cols, axis=1)
        print("Done")

        # One-hot encode for the labels
        print("One-hot encoding the labels ... ", end="")
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

        # Store the scaled data in a file
        print("Store the preprocessed data ... ", end="")
        preprocessed_data = dataframe.to_numpy()
        np.save(target_file, preprocessed_data)
        print("Done")

    return preprocessed_data


# A function to apply PCA to the data
def apply_pca(features, target_file):

    # If the PCA data is already available then read it
    if os.path.isfile(target_file + ".npy"):
        print("PCA data already available")
        print("Read PCA data ... ", end="")
        pca_features = np.load(target_file + ".npy", allow_pickle=True)
        print("Done")

    # Dimension reduction using PCA
    else:
        print("Applying PCA to preprocessed data ... ", end="")
        pca = PCA()
        pca.fit(features)
        pca_features = pca.transform(features)
        print("Done")
        print("Storing the PCA data in a file ... ", end="")
        np.save(target_file, pca_features)
        print("Done")

    return pca_features


# A function to apply LDA to the data
def apply_lda(features, y, target_file):

    # If the LDA data is already available then read it
    if os.path.isfile(target_file + ".npy"):
        print("LDA data already available")
        print("Read LDA data ... ", end="")
        lda_features = np.load(target_file + ".npy", allow_pickle=True)
        print("Done")

    # Dimension reduction using LDA
    else:
        print("Applying LDA to preprocessed data ... ", end="")
        lda = LinearDiscriminantAnalysis()
        lda.fit(features, y)
        lda_features = lda.transform(features)
        print("Done")
        print("Storing the LDA data in a file ... ", end="")
        np.save(target_file, lda_features)
        print("Done")

    return lda_features


# A function to apply UMAP to the data
def apply_umap(features, dimensions, random_state, target_file):

    # If the UMAP data is already available then read it
    if os.path.isfile(target_file + ".npy"):
        print("UMAP data for " + str(dimensions) + " dimensions already available")
        print("Read UMAP data for " + str(dimensions) + " dimensions ... ", end="")
        umap_features = np.load(target_file + ".npy", allow_pickle=True)
        print("Done")

    else:
        # Dimension reduction using UMAP
        print("Applying UMAP (" + str(dimensions) + " dimensions) to preprocessed data ... ", end="")
        # Random state argument is used for reproducibility
        ump = UMAP(n_components=dimensions, init="random", random_state=random_state, min_dist=0)
        ump.fit(features)
        umap_features = ump.transform(features)
        print("Done")
        print("Storing the UMAP data (" + str(dimensions) + " dimensions) in a file ... ", end="")
        np.save(target_file, umap_features)
        print("Done")

    return umap_features
