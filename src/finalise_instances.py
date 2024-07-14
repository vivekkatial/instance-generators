import pandas as pd
import os

def main():

    # Define experiment
    experiment = 'INFORMS-Revision-12-node-network'

    # Create directory for instances to run on SPARTAN
    instances_dir = f'{experiment}/instances_final/'

    # Check if directory exists
    if not os.path.exists(instances_dir):
        os.makedirs(instances_dir)
    
    # Read in the instance data
    instance_data = pd.read_csv(f'{experiment}/final_evolved_instances_n_12_with_source.csv')


    # Create new filenames for the instances that includes their source and removes the 
    # final_population_n_12/ prefix
    instance_data['new_filename'] = instance_data['Source']

    # Copy the files to the new directory based on filename and new filename
    for index, row in instance_data.iterrows():
        # Append final_population_n_12 to the filename
        filename = experiment + '/best_graphs_12/' + row['Source']
        new_filename = row['new_filename']
        # Copy the file to the new directory
        print(f'Copying {filename} to {instances_dir}{new_filename}')
        os.system(f'cp {filename} {instances_dir}{new_filename}')

        


if __name__ == '__main__':
    main()