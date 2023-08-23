# import pandas as pd
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans

# def split_data_to_classes(input_csv, output_csv, use_pca=True):
#     # Load the data from the input CSV
#     All_Data_Training = pd.read_csv(input_csv)
    
#     # Extract distance columns
#     distance_columns = All_Data_Training.loc[:, 'resname GTP':'resid 29']

#     if use_pca:
#         # Use PCA to reduce dimensionality to three columns
#         pca = PCA(n_components=3)
#         distance_pca = pca.fit_transform(distance_columns)
#         distance_for_clustering = distance_pca
#     else:
#         distance_for_clustering = distance_columns

#     num_clusters = 2
#     kmeans = KMeans(n_clusters=num_clusters, random_state=42)
#     kmeans.fit(distance_for_clustering)

#     cluster_labels = kmeans.labels_

#     All_Data_Training['cluster_label'] = cluster_labels

#     class_0_files = All_Data_Training[All_Data_Training['cluster_label'] == 0]['files'].tolist()
#     class_1_files = All_Data_Training[All_Data_Training['cluster_label'] == 1]['files'].tolist()

#     class_0_data = All_Data_Training[All_Data_Training['cluster_label'] == 0]
#     class_1_data = All_Data_Training[All_Data_Training['cluster_label'] == 1]

#     # print("Files in Class 0:")
#     # print(class_0_files)

#     # print("Files in Class 1:")
#     # print(class_1_files)
    
#     # Save the modified data with class labels to a new CSV file
#     All_Data_Training.to_csv(output_csv, index=False)
#     class_0_data.to_csv('./data/class_0_data.csv', index=False)
#     class_1_data.to_csv('./data/class_1_data.csv', index=False)


# if __name__ == "__main__":
#     input_csv = './data/All_Data_Training.csv'  # Update input file path
#     output_csv = './data/All_Data_Training_with_classes.csv'  # Update output file path
#     use_pca = True  # Set to False to compare without PCA
#     split_data_to_classes(input_csv, output_csv, use_pca)   

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Constants
input_csv = './data/All_Data_Training.csv'
output_csv = './data/All_Data_Training_with_classes.csv'
class_0_data_csv = './data/class_0_data.csv'
class_1_data_csv = './data/class_1_data.csv'
use_pca = True

def split_data_to_classes(input_csv, output_csv, use_pca=True):
    """
    Split data into classes using clustering and save modified data with class labels to CSV files.

    Parameters:
    ----------
    input_csv : str
        Path to the input CSV file.
    output_csv : str
        Path to save the modified data with class labels.
    use_pca : bool, optional
        Whether to use PCA for dimensionality reduction, by default True
    """
    logging.info("Loading data from the input CSV...")
    all_data_training = pd.read_csv(input_csv)

    distance_columns = all_data_training.loc[:, 'resname GTP':'resid 29']

    if use_pca:
        logging.info("Performing PCA for dimensionality reduction...")
        pca = PCA(n_components=3)
        distance_pca = pca.fit_transform(distance_columns)
        distance_for_clustering = distance_pca
    else:
        distance_for_clustering = distance_columns

    num_clusters = 2
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(distance_for_clustering)

    cluster_labels = kmeans.labels_
    all_data_training['cluster_label'] = cluster_labels

    class_0_files = all_data_training[all_data_training['cluster_label'] == 0]['files'].tolist()
    class_1_files = all_data_training[all_data_training['cluster_label'] == 1]['files'].tolist()

    class_0_data = all_data_training[all_data_training['cluster_label'] == 0]
    class_1_data = all_data_training[all_data_training['cluster_label'] == 1]

    logging.info("Saving modified data to CSV files...")
    all_data_training.to_csv(output_csv, index=False)
    class_0_data.to_csv(class_0_data_csv, index=False)
    class_1_data.to_csv(class_1_data_csv, index=False)

def main():
    split_data_to_classes(input_csv, output_csv, use_pca)

if __name__ == "__main__":
    main()


