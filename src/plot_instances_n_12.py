import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Polygon
from tqdm import tqdm
from src.data.targets_evolved import target_points

def load_and_prepare_instance_data(filepath, source_name, gen_pattern):
    df = pd.read_csv(filepath)
    df['Generation'] = df['Source'].str.extract(gen_pattern).astype(int)
    df['Filename'] = df['Source']
    df['Source'] = source_name
    df['Population Type'] = source_name
    return df

def is_close_to_any_other_point(row, df, tolerance=0.1):
    """
    Check if the given row's z_1 and z_2 values are within +/- tolerance of any other point in the DataFrame.

    :param row: The row of the DataFrame to check.
    :param df: The entire DataFrame.
    :param tolerance: The tolerance within which to check proximity.
    :return: True if close to any other point, False otherwise.
    """
    for _, other_row in df.iterrows():
        # Ensure we're not comparing the row to itself
        if row.name != other_row.name:
            if abs(row['z_1'] - other_row['z_1']) <= tolerance and abs(row['z_2'] - other_row['z_2']) <= tolerance:
                return True
    return False

def main():
    experiment = 'qaoa-param-evolved'

    print('=========================================================================')
    print(f'-> Plotting Instances from {experiment}.')
    print('=========================================================================')

    paths_and_sources = [
        (os.path.join('final_population_n_12', 'new-instance-coordinates.csv'), 'Evolved Population (n=12)', r'_(\d+)\.graphml$'), 
    ]

    # Load and prepare all instance data
    new_instances = pd.concat([load_and_prepare_instance_data(path, source, pattern) for path, source, pattern in paths_and_sources])
    total_instances = new_instances.shape[0]

    print(f"Total number of instances: {total_instances}")
    # Remove overlapping instances
    new_instances = new_instances.drop_duplicates(subset=['z_1', 'z_2'])

    # # Log total number of instances removed from overlap
    print(f"Number of instances removed from overlap: {total_instances - new_instances.shape[0]}")

    tqdm.pandas(desc=f"Checking points {new_instances.shape[0]} instances for proximity to other points")
    # Only apply the function to the evolved instances
    new_instances['close_to_other_point'] = new_instances.progress_apply(lambda row: is_close_to_any_other_point(row, new_instances), axis=1)
    new_instances = new_instances[~new_instances['close_to_other_point']]

    # Log total number of instances removed from overlap and proximity
    print(f"Number of instances removed from overlap: {total_instances - new_instances.shape[0]}")


    # Load the original data and bounds
    data = pd.read_csv(f'{experiment}/coordinates.csv')
    # data['Source'] = data['Row'].str.extract(r'_(\w+)$').str.title().str.replace('_', ' ')
    data['Source'] = data['Row'].str.extract(r'_(\w+)$')
    # Make source title case and remove underscores
    data['Source'] = data['Source'].str.title().str.replace('_', ' ')
    data['Population Type'] = 'Original Instances'
    # Load the bounds data
    bounds = pd.read_csv(f'{experiment}/bounds_prunned.csv')

    # Set plot aesthetics
    sns.set(style="whitegrid", context="paper")
    plt.figure(figsize=(10, 10))
    plt.rcParams['figure.dpi'] = 300

    # Plot boundary
    boundary_points = bounds[['z_1', 'z_2']].values
    plt.gca().add_patch(Polygon(boundary_points, closed=True, fill=False, edgecolor='r', linewidth=2))

    result_df = pd.concat([new_instances, data], ignore_index=True)
    # Define colors and markers
    colors = {
        "Original Instances": "grey",
        "Evolved Population (n=12)": "brown",

    }
    markers = {
        "Original Instances": "o",  # Circle
        "Evolved Population (n=12)": "D",  # Diamond
    }


    plt.figure(figsize=(10, 8))

    # Filter out rows with three highest values of z_2 and row with lowest value of z_1
    # sort by z_2
    result_df = result_df.sort_values(by='z_2', ascending=False)
    result_df = result_df.iloc[3:]
    # sort by z_1
    result_df = result_df.sort_values(by='z_1', ascending=True)
    result_df = result_df.iloc[1:]




    # Plot each group with its respective color and marker
    for population_type, group_df in result_df.groupby('Population Type'):
        # If the population type is 'Original Instances', alpha is 0.3, else 1
        alpha = 0.3 if population_type == 'Original Instances' else 0.8
        plt.scatter(group_df['z_1'], group_df['z_2'], 
                    color=colors[population_type], 
                    marker=markers[population_type], 
                    label=population_type, 
                    alpha=alpha)
    
    # Plot the target point
    for target_point in target_points:
        plt.scatter(target_point[0], target_point[1], marker='x', color='black', s=60)


    # Plot boundary
    boundary_points = bounds[['z_1', 'z_2']].values
    plt.gca().add_patch(Polygon(boundary_points, closed=True, fill=False, edgecolor='r', linewidth=2))


    # Set plot aesthetics
    plt.title("Evolved Instances")

    # Print out stats about the evolved instances
    print(f"Number of evolved instances: {new_instances.shape[0]}")
    print(f"Number of original instances: {data.shape[0]}")
    




    plt.xlabel("$z_1$")
    plt.ylabel("$z_2$")
    plt.legend(title="Population Type")
    plt.grid(True)

    # Write the plot to a file
    plt.savefig('evolved_instances_n_12.png')
    # Save evolved instances to a filen
    new_instances.to_csv(f'{experiment}/final_evolved_instances_n_12.csv', index=False)

if __name__ == "__main__":
    main()
