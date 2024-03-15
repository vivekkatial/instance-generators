import pandas as pd
import os

def main():

    # Create directory for instances to run on SPARTAN
    instances_dir = 'instances_final/'

    # Check if directory exists
    if not os.path.exists(instances_dir):
        os.makedirs(instances_dir)
    
    # Read in the instance data
    instance_data = pd.read_csv('data/final_evolved_instances_n_12_with_source.csv')

    # Convert source to snake_case
    instance_data['Source'] = instance_data['Source'].str.lower().str.replace(' ', '_')

    # Create new filenames for the instances that includes their source and removes the 
    # final_population_n_12/ prefix
    instance_data['new_filename'] = instance_data['Source'] + '_' + instance_data['Filename'].str.replace('final_population_n_12/', '')

    # Copy the files to the new directory based on filename and new filename
    for index, row in instance_data.iterrows():
        # Append final_population_n_12 to the filename
        filename = 'final_population_n_12/' + row['Filename']
        new_filename = row['new_filename']
        # Copy the file to the new directory
        print(f'Copying {filename} to {instances_dir}{new_filename}')
        os.system(f'cp {filename} {instances_dir}{new_filename}')


if __name__ == '__main__':
    main()