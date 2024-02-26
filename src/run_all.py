import subprocess
from concurrent.futures import ProcessPoolExecutor
from targets import target_points

# Define the paths to your scripts
scripts = [
    'src/genetic_algorithm.py',
    'src/preprocess.py',
    'src/plot_instances.py'
]

# Define a function to execute all scripts sequentially for a given target_point
def run_scripts_sequentially(target_point):
    for script in scripts:
        subprocess.run(['poetry', 'run', 'python', script, '--target_point'] + target_point, check=True)

# Use ProcessPoolExecutor to parallelize the execution across target_points
def main():
    with ProcessPoolExecutor() as executor:
        # Submit a task for each target_point to run all scripts sequentially for that target_point
        futures = [executor.submit(run_scripts_sequentially, target_point) for target_point in target_points]
        
        # Wait for all futures to complete (optional, if you need to process results)
        for future in futures:
            try:
                future.result()  # This will re-raise any exception that occurred during the execution
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()

