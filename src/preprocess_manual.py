import numpy as np
import networkx as nx
import os
import pandas as pd
import scipy.io
import warnings
from scipy import stats

# Custom imports (assuming these modules are defined elsewhere in your project)
from graph_features import build_feature_df
from graph_instance import create_graphs_from_all_sources, GraphInstance

def check_files_exist():
    """Check if the necessary files exist."""
    try:
        with open('data/metadata.csv') as f:
            pass
        with open('data/precomputed-min-vals.csv') as f:
            pass
        with open('data/model.mat') as f:
            pass
        with open('data/projection_matrix.csv') as f:
            pass
    except FileNotFoundError:
        return False
    return True

def ignore_warnings():
    """Ignore specific warnings."""
    warnings.filterwarnings("ignore", category=Warning)

def load_min_values(filepath: str) -> dict:
    """Load precomputed minimum values from a CSV file."""
    min_vals = pd.read_csv(filepath, index_col=0)
    return min_vals.iloc[0].to_dict()

def try_convert_numeric(s):
    """Attempt to convert a Series to numeric, handling errors explicitly."""
    try:
        return pd.to_numeric(s, downcast='float')
    except ValueError:
        return s

def preprocess_features(df: pd.DataFrame, min_vals: dict) -> pd.DataFrame:
    """Preprocess features by converting types and adjusting with min_vals."""
    # Convert object and bool columns to numeric, handling errors explicitly
    for col in df.select_dtypes(include=['object', 'bool']).columns:
        df[col] = try_convert_numeric(df[col])
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    # Adjust numeric columns with min_vals
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col] + abs(min_vals.get(col, 0))
    return df


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract feature matrix from the DataFrame."""
    return df.iloc[:, 1:25].values

def apply_preprocessing(X: np.ndarray, mat_contents: dict) -> np.ndarray:
    """Apply preprocessing steps including box-cox transformation and z-transform."""
    lambdaX, out_minX, muX, sigmaX = (
        mat_contents['prelim'][0]['lambdaX'][0][0],
        mat_contents['prelim'][0]['minX'][0],
        mat_contents['prelim'][0]['muX'][0],
        mat_contents['prelim'][0]['sigmaX'][0]
    )

    X = X - out_minX + 1
    for i in range(X.shape[1]):
        X[:, i] = stats.boxcox(X[:, i], lmbda=lambdaX[i])
    X = (X - muX) / sigmaX

    feat_indices = mat_contents["featsel"][0][0][0][0] - 1  # Adjust for Python indexing
    return X[:, feat_indices]

def project_features(X: np.ndarray, projection_matrix_file: str) -> pd.DataFrame:
    """Project features to a lower-dimensional space."""
    proj_matrix = pd.read_csv(projection_matrix_file, index_col=0)
    projection_matrix_np = proj_matrix.values.T
    z_coor = X @ projection_matrix_np
    return pd.DataFrame(z_coor, columns=['z_1', 'z_2'])

def get_z1_z2_projection(graph):
    """Get the Z1, Z2 projection of a single graph."""
    # Custom function to get the Z1, Z2 projection of an arbitrary graph
    ignore_warnings()
    check_files_exist()
    
    min_vals = load_min_values('data/precomputed-min-vals.csv')
    df = pd.read_csv('data/metadata.csv', index_col=0, nrows=0)    
    features = build_feature_df(graph, "generated-instance")
    df = pd.concat([df, pd.DataFrame([features])], ignore_index=True)
    df = preprocess_features(df, min_vals)
    X = build_feature_matrix(df)
    mat_contents = scipy.io.loadmat('data/model.mat')
    X = apply_preprocessing(X, mat_contents)
    new_instance_projections = project_features(X, 'data/projection_matrix.csv')
    return new_instance_projections.iloc[0,:].values

def read_graphs_from_pickles(pickle_directory_path):
    # List to store the graphs
    graphs = []
    
    # List all files in the given directory
    for filename in os.listdir(pickle_directory_path):
        if filename.endswith(".pkl"):
            # Construct full file path
            file_path = os.path.join(pickle_directory_path, filename)
            # Read the graph from the pickle file
            graph = nx.read_gpickle(file_path)
            # Add the graph to the list
            graphs.append(graph)
            print(f"Loaded graph from {filename} with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    
    return graphs


def main():
    print('=========================================================================')
    print('-> Apply Preprocessing.')
    print('=========================================================================')
    
    ignore_warnings()
    check_files_exist()
    
    min_vals = load_min_values('data/precomputed-min-vals.csv')
    df = pd.read_csv('data/metadata.csv', index_col=0, nrows=0)
    graphs = read_graphs_from_pickles('best_graphs/')
    
    for i, inst in enumerate(graphs):
        # features = build_feature_df(inst, "GA-generated-instance")
        features = build_feature_df(inst, "GA-generated-instance_" + str(i))
        df = pd.concat([df, pd.DataFrame([features])], ignore_index=True)
    
    df = preprocess_features(df, min_vals)
    X = build_feature_matrix(df)
    print('-> Bounding outliers, scaling, and normalizing the data.')
    
    mat_contents = scipy.io.loadmat('data/model.mat')
    X = apply_preprocessing(X, mat_contents)
    
    new_instance_projections = project_features(X, 'data/projection_matrix.csv')
    new_instance_projections['Source'] = df['Source']

    # Remove NaNs
    new_instance_projections = new_instance_projections.dropna()
    new_instance_projections.to_csv('data/new-instance-coordinates.csv', index=False)
    print(new_instance_projections)

if __name__ == "__main__":
    main()
