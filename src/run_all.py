import subprocess
import os
from concurrent.futures import ProcessPoolExecutor
from targets import target_points

# Define the paths to your scripts
scripts = [
    'src/genetic_algorithm.py',
    'src/preprocess.py',
    'src/plot_instances.py'
]

# First check how many directories for each target_point already exist
def check_existing_directories(target_point):
    # Convert the float coordinates in target_point to strings
    str_target_point = list(map(str, target_point))
    # Create the directory name for the target_point
    directory_name = f"target-point-graphs/target_point_{str_target_point[0]}_{str_target_point[1]}_n_50"
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
    # Set max_workers to the number of available cores less two
    max_workers = os.cpu_count() - 2
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Only run for target_points that do not have a directory yet
        # target_points = [target_point for target_point in target_points if not check_existing_directories(target_point)]
        # Submit a task for each target_point to run all scripts sequentially for that target_point
        futures = [executor.submit(run_scripts_sequentially, target_point) for target_point in target_points]
        
        # Wait for all futures to complete (optional, if you need to process results)
        for future in futures:
            try:
                future.result()  # This will re-raise any exception that occurred during the execution
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
