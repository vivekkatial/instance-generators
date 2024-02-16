import numpy as np
import warnings
import pandas as pd
import networkx as nx
import scipy.io
import numpy as np

from graph_features import get_graph_features

# Ignore FutureWarnings specifically from the laplacian_matrix function
warnings.filterwarnings("ignore", category=FutureWarning, 
                        message="laplacian_matrix will return a scipy.sparse array instead of a matrix in Networkx 3.0.")


def generate_new_instance(num_nodes=12, prob=0.5):
    """
    Generates a new graph instance and extracts its features.
    
    Parameters:
    - num_nodes: int, number of nodes in the graph.
    - prob: float, probability for edge creation.
    
    Returns:
    - features: dict, features of the generated graph.
    """
    # Generate a random graph
    G_1 = nx.random_regular_graph(d=3,n=num_nodes)
    G_2 = nx.erdos_renyi_graph(num_nodes, prob)
    source_1 = 'New Regular Graph'
    source_2 = 'New Random Graph'

    # Randomly pick one of the two graphs
    if np.random.rand() > 0.5:
        G = G_1
        source = source_1
    else:
        G = G_2
        source = source_2

    return G, source

def build_feature_df(G, source):
    
    graph_features = get_graph_features(G)

    # Map the features to the ISA metadata csv
    mapped_features = {
    'Source': source,
    'feature_density': graph_features['density'],
    'feature_radius': graph_features['radius'],
    'feature_minimum_degree': graph_features['minimum_degree'],
    'feature_n_layers': 4,
    'feature_algebraic_connectivity': graph_features['algebraic_connectivity'],
    'feature_connected': graph_features['connected'],
    'feature_number_of_cut_vertices': graph_features['number_of_cut_vertices'],
    'feature_minimum_dominating_set': graph_features['minimum_dominating_set'],
    'feature_diameter': graph_features['diameter'],
    'feature_laplacian_second_largest_eigenvalue': graph_features['laplacian_second_largest_eigenvalue'],
    'feature_number_of_components': graph_features['number_of_components'],
    'feature_smallest_eigenvalue': graph_features['smallest_eigenvalue'],
    'feature_regular': graph_features['regular'],
    'feature_planar': graph_features['planar'],
    'feature_bipartite': graph_features['bipartite'],
    'feature_clique_number': graph_features['clique_number'],
    'feature_eulerian': graph_features['eulerian'],
    'feature_average_distance': graph_features['average_distance'],
    'feature_edge_connectivity': graph_features['edge_connectivity'],
    'feature_maximum_degree': graph_features['maximum_degree'],
    'feature_vertex_connectivity': graph_features['vertex_connectivity'],
    'feature_laplacian_largest_eigenvalue': graph_features['laplacian_largest_eigenvalue'],
    'feature_number_of_orbits': graph_features['number_of_orbits'],
    'feature_ratio_of_two_largest_laplacian_eigenvaleus': graph_features['ratio_of_two_largest_laplacian_eigenvaleus'],
    'feature_group_size': graph_features['group_size'],
    'feature_number_of_edges': graph_features['number_of_edges'],
    'feature_number_of_minimal_odd_cycles': graph_features['number_of_minimal_odd_cycles'],
    'algo_instance_class_optimsed': 0,
    'algo_random_initialisation': 0,
    'algo_three_regular_graph_optimised': 0,
    'algo_tqa_initialisation': 0
    }

    return mapped_features


def project_and_map(features, proj_matrix_path):
    """
    Projects the features onto a new space using the provided projection matrix and maps to result vector.
    
    Parameters:
    - features: dict, features of the graph.
    - proj_matrix_path: str, path to the projection matrix CSV file.
    
    Returns:
    - result_vector: numpy.ndarray, the projected result vector.
    """
    # Import the projection matrix
    proj_matrix = pd.read_csv(proj_matrix_path, index_col=0)
    
    # Extract the projection matrix column names
    colnames = proj_matrix.columns
    
    # Check that all features in the projection matrix are also in the feature dictionary
    for col in colnames:
        if col not in features.keys():
            raise ValueError(f'Feature {col} not in feature dictionary.')
    
    # Extract feature values in the order of the projection matrix columns
    feature_values = [features[col] for col in colnames if col in features]
    
    # Convert to numpy array and reshape for multiplication
    feature_vector = np.array(feature_values).reshape(1, -1)
    
    # Convert projection matrix to numpy array and transpose
    projection_matrix_np = proj_matrix.values.T
    
    # Perform the matrix multiplication
    result_vector = feature_vector @ projection_matrix_np
    return result_vector

# Example usage
print('=========================================================================')
print('-> Auto-pre-processing.')
print('=========================================================================')
# Build empty df from metadata
df = pd.read_csv('data/metadata.csv', index_col=0, nrows=0)

# Generate 10 new instances
for i in range(10):
    G, source = generate_new_instance(12, 0.5)
    features = build_feature_df(G, source)
    df = pd.concat([df, pd.DataFrame([features])], ignore_index=True)

import pdb; pdb.set_trace()


# Load the .mat file
mat_contents = scipy.io.loadmat('data/model.mat')

# Load the transform data
medvalues = mat_contents['prelim'][0]['medval']
hibound = mat_contents['prelim'][0]['hibound']
lobound = mat_contents['prelim'][0]['lobound']
minX = mat_contents['prelim'][0]['minX']
lambdaX = mat_contents['prelim'][0]['lambdaX']
sigmaX = mat_contents['prelim'][0]['sigmaX']


# Project and map to result vector
result_vector = project_and_map(features, 'data/projection_matrix.csv')
print("Resulting vector (z_1, z_2):", result_vector)

# Select columns 2 to 29 from df
features = df.iloc[:, 2:29]