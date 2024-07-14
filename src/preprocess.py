import numpy as np
import networkx as nx
import os
import pandas as pd
import scipy.io
import warnings
import argparse

from scipy import stats

# Custom imports (assuming these modules are defined elsewhere in your project)
from src.graph_features import build_feature_df
from src.graph_instance import create_graphs_from_all_sources, GraphInstance


def check_files_exist(experiment="data"):
    """Check if the necessary files exist."""
    try:
        with open(f'{experiment}/metadata.csv') as f:
            pass
        with open(f'{experiment}/precomputed-min-vals.csv') as f:
            pass
        with open(f'{experiment}/model.mat') as f:
            pass
        with open(f'{experiment}/projection_matrix.csv') as f:
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

    # Adjust numeric columns with min_vals (only if they exist)
    if min_vals is None:
        return df
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col] + abs(min_vals.get(col, 0))
    return df


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract feature matrix from the DataFrame."""
    # Select columns that start with feature
    df = df.filter(regex='feature')
    return df.values


def apply_preprocessing(X: np.ndarray, mat_contents: dict) -> np.ndarray:
    """Apply preprocessing steps including box-cox transformation and z-transform."""
    lambdaX, out_minX, muX, sigmaX = (
        mat_contents['prelim'][0]['lambdaX'][0][0],
        mat_contents['prelim'][0]['minX'][0],
        mat_contents['prelim'][0]['muX'][0],
        mat_contents['prelim'][0]['sigmaX'][0],
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


def get_z1_z2_projection(graph, **kwargs):
    """Get the Z1, Z2 projection of a single graph."""

    # Check if experiment exists in kwargs as a param
    if "experiment" in kwargs:
        experiment = kwargs["experiment"]
    else:
        experiment = "data"


    # Custom function to get the Z1, Z2 projection of an arbitrary graph
    ignore_warnings()
    check_files_exist(experiment=experiment)

    # Check if precomputed_min needed (only if the file exists)
    if os.path.exists(f"{experiment}/precomputed-min-vals.csv"):
        min_vals = load_min_values(f"{experiment}/precomputed-min-vals.csv")
    else:
        min_vals = None

    df = pd.read_csv(f'{experiment}/metadata.csv', index_col=0, nrows=0)
    features = build_feature_df(graph, "generated-instance")
    df = pd.concat([df, pd.DataFrame([features])], ignore_index=True)
    df = preprocess_features(df, min_vals)
    X = build_feature_matrix(df)
    mat_contents = scipy.io.loadmat(f'{experiment}/model.mat')
    X = apply_preprocessing(X, mat_contents)
    new_instance_projections = project_features(X, f'{experiment}/projection_matrix.csv')
    return new_instance_projections.iloc[0, :].values


def read_graphs_from_graphml(graphml_directory_path):
    # List to store the graphs
    graphs = []

    graphs_missing = 0
    # List all files in the given directory
    for filename in os.listdir(graphml_directory_path):
        if filename.endswith(".graphml"):
            try:
                # Construct full file path
                file_path = os.path.join(graphml_directory_path, filename)
                # Read the graph from the graphml
                graph = nx.read_graphml(file_path)
                # Add the graph to the list along with its name
                graphs.append((graph, filename))
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                graphs_missing += 1
                # Delete that target point directory
                os.remove(file_path)
                
    if graphs_missing > 0:
        print(f"Warning: {graphs_missing} graphs could not be read.")
                


    return graphs


def main():
    print('=========================================================================')
    print('-> Apply Preprocessing.')
    print('=========================================================================')

    ignore_warnings()
    check_files_exist()

    # Load argument
    parser = argparse.ArgumentParser(description='Process target point')
    parser.add_argument(
        "--target_point",
        type=float,
        nargs=2,
        default=[2.5, 2.5],
        help="The target point for the fitness function.",
    )

    # Add bool argument for best graphs
    parser.add_argument(
        "--best_graphs_n_12",
        type=bool,
        default=False,
        help="Whether to use best graphs or not.",
    )

    # Add bool argument for best graphs
    parser.add_argument(
        "--best_graphs_n_16",
        type=bool,
        default=False,
        help="Whether to use best graphs or not.",
    )

    # Add bool argument for best graphs
    parser.add_argument(
        "--best_graphs_n_20",
        type=bool,
        default=False,
        help="Whether to use best graphs or not.",
    )

    # Add bool argument for best graphs n_24
    parser.add_argument(
        "--best_graphs_n_24",
        type=bool,
        default=False,
        help="Whether to use best graphs or not.",
    )

    parser.add_argument(
        "--best_graphs_n_50",
        type=bool,
        default=False,
        help="Whether to use best graphs or not.",
    )

    parser.add_argument(
        "--final_n_12",
        type=bool,
        default=False,
        help="Whether to use best graphs or not.",
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default='INFORMS-Revision-12-node-network',
        help="The name of the experiment being conducted.",
    )

    parser.add_argument(
        "--node_size",
        type=int,
        default=None,
        help="The size of the nodes in the graph.",
    )

    parser.add_argument(
        "--all_instances",
        type=bool,
        default=False,
        help="Whether to use all instances or not.",
    )

    args = parser.parse_args()
    target_point = args.target_point
    best_graphs_n_12 = args.best_graphs_n_12
    best_graphs_n_16 = args.best_graphs_n_16
    best_graphs_n_20 = args.best_graphs_n_20
    best_graphs_n_24 = args.best_graphs_n_24
    best_graphs_n_50 = args.best_graphs_n_50
    all_instances = args.all_instances
    node_size = args.node_size
    final_n_12 = args.final_n_12
    experiment = args.experiment

    # Only one of the three options can be True
    if sum([best_graphs_n_12, best_graphs_n_16, best_graphs_n_24, best_graphs_n_50]) > 1:
        raise ValueError("Only one of the three options can be True")
    
    if best_graphs_n_12:
        load_path = "best_graphs_12/"
    elif best_graphs_n_16:
        load_path = "best_graphs_16/"
    elif best_graphs_n_20:
        load_path = "best_graphs_20/"
    elif best_graphs_n_24:
        load_path = "best_graphs_24/"
    elif best_graphs_n_50:
        load_path = "best_graphs_50/"
    elif final_n_12:
        load_path = "final_population_n_12/"
    elif node_size is not None and experiment is not None and not all_instances:
        load_path = os.path.join(experiment, f"best_graphs_{node_size}")
    elif node_size is not None and experiment is not None and all_instances:
        load_path = os.path.join(experiment, f"all-evolved-instances")
    else:
        load_path = os.path.join(experiment, "target-point-graphs", f"target_point_{target_point[0]}_{target_point[1]}_n_12")

    # Check if precomputed_min needed (only if the file exists)
    if os.path.exists(f'{experiment}/precomputed-min-vals.csv'):
        min_vals = load_min_values(f'{experiment}/precomputed-min-vals.csv')
    else:
        min_vals = None
    df = pd.read_csv(f'{experiment}/metadata.csv', index_col=0, nrows=0)
    graphs = read_graphs_from_graphml(load_path)
    # Get each graphs filename
    filenames = [inst[1] for inst in graphs]

    for i, inst in enumerate(graphs):
        features = build_feature_df(inst[0], inst[1])
        df = pd.concat([df, pd.DataFrame([features])], ignore_index=True)

    df = preprocess_features(df, min_vals)

    X = build_feature_matrix(df)
    print('-> Bounding outliers, scaling, and normalizing the data.')

    mat_contents = scipy.io.loadmat(f'{experiment}/model.mat')
    X = apply_preprocessing(X, mat_contents)

    new_instance_projections = project_features(X, f'{experiment}/projection_matrix.csv')
    new_instance_projections['Source'] = df['Source']

    # Remove NaNs
    new_instance_projections = new_instance_projections.dropna()
    # Save the new instance projections to a CSV file
    new_inst_file = load_path + "/new-instance-coordinates.csv"
    new_instance_projections.to_csv(new_inst_file, index=False)
    print(new_instance_projections)



if __name__ == "__main__":
    main()
