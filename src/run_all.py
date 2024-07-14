import subprocess
import os
from concurrent.futures import ProcessPoolExecutor
from src.data.targets_evolved import target_points

EXPERIMENT='INFORMS-Revision-12-node-network'

# Define the paths to your scripts
scripts = [
    'src/algorithm/ga.py',
    'src/preprocess.py',
    'src/plot_instances.py'
]

# First check how many directories for each target_point already exist
def check_existing_directories(target_point):
    # Convert the float coordinates in target_point to strings
    str_target_point = list(map(str, target_point))
    # Create the directory name for the target_point
    directory_name = f"{EXPERIMENT}/target-point-graphs/target_point_{str_target_point[0]}_{str_target_point[1]}_n_14"
    # Check if graph_features.json exists in the directory
    return os.path.exists(os.path.join(directory_name, 'graph_features.json'))

# Define a function to execute all scripts sequentially for a given target_point
def run_scripts_sequentially(target_point):
    # Convert the float coordinates in target_point to strings
    str_target_point = list(map(str, target_point))
    for script in scripts:
        # Pass the target_point as strings to the subprocess.run function
        subprocess.run(['poetry', 'run', 'python', script, '--target_point'] + str_target_point, check=True)

# Use ProcessPoolExecutor to parallelize the execution across target_points
def main(target_points):
    # Set max_workers to the number of available cores less four
    max_workers = os.cpu_count() - 2
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Only run for target_points that do not have a directory yet
        # target_points_to_process = [target_point for target_point in target_points if not check_existing_directories(target_point)]
        target_points_to_process = target_points
        # Print the target_points that will be processed
        print(f"Processing {len(target_points_to_process)} target points: {target_points_to_process}")
        # Count the number of target_points that already have a directory
        existing_targets = len(target_points) - len(target_points_to_process)
        print(f"{existing_targets} target points already have a directory.")
        # Submit a task for each target_point to run all scripts sequentially for that target_point
        futures = [executor.submit(run_scripts_sequentially, target_point) for target_point in target_points_to_process]
        
        # Wait for all futures to complete (optional, if you need to process results)
        for future in futures:
            try:
                future.result() 
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == '__main__':
    import time
    # START TIMER
    start = time.time()
    main(target_points)
    # END TIMER
    end = time.time()
    # Print the time elapsed and round to 2 decimal places
    print(f"Time elapsed: {round(end - start, 2)} seconds")
