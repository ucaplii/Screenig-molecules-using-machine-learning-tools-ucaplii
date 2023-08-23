import os
import csv
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from natsort import natsorted
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def generate_morgan_fingerprints(ligand_dir, output_file):
    """
    Generate Morgan fingerprints from PDB files in the specified input directory and save them to the output file.

    Parameters:
    ----------
    input_dir : str
        Path to the directory containing PDB files.
    output_file : str
        Path to the output CSV file to save the Morgan fingerprints.
    """
    mol_list = []
    for ligand_filename in natsorted(os.listdir(ligand_dir)):
        ligand_filepath = os.path.join(ligand_dir, ligand_filename)
        if ligand_filepath.endswith('.pdb'):
            mol = Chem.MolFromPDBFile(ligand_filepath)
            mol_list.append(mol)

    morgan_fingerprints = []
    for mol in mol_list:
        fingerprint = AllChem.GetHashedMorganFingerprint(mol, 1, nBits=1024)
        fingerprint_array = np.zeros((0,), dtype=int)
        DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
        morgan_fingerprints.append(fingerprint_array)

    with open(output_file, 'w', newline='') as csvfile:
        
        writer = csv.writer(csvfile)
        header = [i for i in range(1, 1025)]
        writer.writerow(header)
        writer.writerows(morgan_fingerprints)

def calculate_ligand_protein_distances(ligand_dir, protein_file, output_file):
    """
    Calculate distances between ligands and specified protein residues and save results to a CSV file.

    Parameters:
    ----------
    ligand_dir : str
        Path to the directory containing ligand PDB files.
    protein_file : str
        Path to the protein PDB file.
    output_file : str
        Path to save the CSV file with calculated distances.
    """
    target_residues = ["resname GTP","resname MG","resid 789", "resid 13", "resid 60", 
                       "resid 59", "resid 16", "resid 76","resid 35","resid 19", 
                       "resid 18", "resid 785", "resid 15","resid 30","resid 14",
                       "resid 117","resid 116","resid 120","resid 119","resid 146",
                       "resid 145", "resid 147","resid 786","resid 61","resid 29"]

    ligand_files = os.listdir(ligand_dir)
    ligand_pdb_files = [f for f in ligand_files if f.endswith('.pdb')]
    num_ligand_pdb_files = len(ligand_pdb_files)

    distance_matrix = np.zeros(shape=(num_ligand_pdb_files, len(target_residues)))
    ligand_index = 0
    for ligand_filename in natsorted(os.listdir(ligand_dir)):
        ligand_filepath = os.path.join(ligand_dir, ligand_filename)
        if ligand_filename.endswith('.pdb'):
            ligand_universe = mda.Universe(ligand_filepath)
            ligand_atoms = ligand_universe.select_atoms('resname LIG')
            ligand_positions = ligand_atoms.positions

            protein_universe = mda.Universe(protein_file)
            for residue_index, target_residue in enumerate(target_residues):
                target_atoms = protein_universe.select_atoms(target_residue)
                target_positions = target_atoms.positions
                dist_array = distances.distance_array(ligand_positions, target_positions)
                shortest_distances = []
                for sublist in dist_array:
                    shortest_index = np.argmin(sublist)
                    shortest_distances.append(sublist[shortest_index])
                min_distance = np.min(shortest_distances)
                distance_matrix[ligand_index, residue_index] = min_distance
            ligand_index += 1

    # Remove columns with all zeros
    distance_matrix = distance_matrix[:, ~np.all(distance_matrix == 0, axis=0)]

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        header = [res for res in target_residues]
        writer.writerow(header)
        writer.writerows(distance_matrix)

def combine_csv_files(distances_file, mf_file, barrier_file, combined_file):
    """
    Combine three CSV files into a single CSV file, reordering columns.

    Parameters:
    ----------
    distances_file : str
        Path to the CSV file with calculated distances.
    mf_file : str
        Path to the CSV file with Morgan fingerprints.
    barrier_file : str
        Path to the CSV file with barrier data.
    combined_file : str
        Path to save the combined CSV file.
    """

    # Read the CSV files into DataFrames
    distances_df = pd.read_csv(distances_file)
    mf_df = pd.read_csv(mf_file)
    barrier_df = pd.read_csv(barrier_file)

    combined_df = pd.concat([barrier_df, mf_df, distances_df], axis=1)
    barriers_col = 'barriers'
    barriers_col_data = combined_df[barriers_col]
    combined_df.drop(columns=[barriers_col], inplace=True)
    # Move the barriers column to the end of the DataFrame
    combined_df[barriers_col] = barriers_col_data


    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(combined_file, index=False)

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
    all_data_training = pd.read_csv(input_csv)

    distance_columns = all_data_training.loc[:, 'resname GTP':'resid 29']

    if use_pca:
        print("Performing PCA for dimensionality reduction...")
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

    
    class_0_data_csv = './data/class_0_data.csv'
    class_1_data_csv = './data/class_1_data.csv'
    all_data_training.to_csv(output_csv, index=False)
    class_0_data.to_csv(class_0_data_csv, index=False)
    class_1_data.to_csv(class_1_data_csv, index=False)


if __name__ == "__main__":

    # Input files and directories
    ligand_directory = './data/ligands'  
    protein_pdb_file = './data/protein/g12d_aligned.pdb'  
    
    MF_csv_file = './data/morgan_fingerprints.csv'  
    distances_csv_file = './data/distances.csv'  

    barriers_csv_file = './data/barriers.csv'  
    combined_csv_file = './data/All_Data_Training.csv'  

    input_csv = './data/All_Data_Training.csv'
    output_csv = './data/All_Data_Training_with_classes.csv'
    use_pca = True
    
    print("")
    print("Genearting input files...")
    print("==========================================================================")
    print("")

    print("Generating Morgan fingerprints...")
    print("==========================================================================")
    generate_morgan_fingerprints(ligand_directory, MF_csv_file)
    print("Morgan fingerprints generation finished.")
    print("")
    
    print("Calculating distances...")
    print("==========================================================================")
    calculate_ligand_protein_distances(ligand_directory, protein_pdb_file, distances_csv_file)
    print("Distances calculation finished.")
    print("")

    print("Combining CSV files...")
    print("==========================================================================")
    combine_csv_files(distances_csv_file, MF_csv_file, barriers_csv_file, combined_csv_file)
    print("CSV files combined.")

    print("Splitting data by clustering...")
    print("==========================================================================")
    split_data_to_classes(input_csv, output_csv, use_pca)
    print("Data split finished.")