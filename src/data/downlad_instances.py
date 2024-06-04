import pandas as pd
import mlflow
import os
import pickle

# Read in runs.csv
runs = pd.read_csv('data/runs.csv')

# Extract the run_id and the instance_class column from runs
evolved_instances = runs[['Run ID', 'instance_class', 'custom_graph']]

# Filter for specific instance classes
instance_classes = [
    "power_law_tree",
    "geometric",
    "nearly_complete_bi_partite",
    "three_regular_graph",
    "uniform_random",
    "watts_strogatz_small_world",
    "four_regular_graph"
]

filtered_instances = evolved_instances[evolved_instances['instance_class'].isin(instance_classes)]

print(filtered_instances.head())

# Create a directory to store the instances
os.makedirs('data/instances', exist_ok=True)

# Set MLflow tracking URI and log in if necessary
mlflow.set_experiment("QAOA-Instance-Based-Parameter-Optimization")

# missing instnaces
missing_instances = []

# Downloading the artifacts
for index, row in filtered_instances.iterrows():
    run_id = row['Run ID']
    instance_class = row['instance_class']
    artifact_path = "graph_instance.pkl"  # Assuming this is the name of the artifact

    # Constructing the artifact's full path in MLflow
    artifact_full_path = os.path.join("runs:", run_id, artifact_path)
    download_file_path = os.path.join("data/instances", f"{run_id}_{instance_class}.pkl")

    try:
        # Check if the file already exists
        if os.path.exists(download_file_path):
            print(f"File {download_file_path} already exists. Skipping download.")
            continue
        # Downloading the artifact
        mlflow.artifacts.download_artifacts(artifact_full_path, dst_path='data/instances')
        # Rename the file to the download_file_path
        os.rename(os.path.join('data/instances', artifact_path), download_file_path)

    except Exception as e:
        print(f"Failed to download artifact for run {run_id} with instance class {instance_class}.")
        missing_instances.append((run_id, instance_class))
        continue



print(f"Missing instances: {missing_instances}")
print("Done downloading instances.")
# Show how many instances are missing
print(f"Number of missing instances: {len(missing_instances)}")

