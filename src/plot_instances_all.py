import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Polygon
from targets import target_points

def load_and_prepare_instance_data(filepath, source_name, gen_pattern):
    df = pd.read_csv(filepath)
    df['Generation'] = df['Source'].str.extract(gen_pattern).astype(int)
    df['Source'] = source_name
    df['Population Type'] = source_name
    return df

def main():
    print('=========================================================================')
    print('-> Plotting Instances.')
    print('=========================================================================')

    paths_and_sources = [
        (os.path.join('best_graphs_12', 'new-instance-coordinates.csv'), 'Evolved Population (n=12)', r'_(\d+)\.pkl$'),
        (os.path.join('best_graphs_16', 'new-instance-coordinates.csv'), 'Evolved Population (n=16)', r'_(\d+)\.pkl$'),
        (os.path.join('best_graphs_24', 'new-instance-coordinates.csv'), 'Evolved Population (n=24)', r'_(\d+)\.pkl$'),
        (os.path.join('best_graphs_50', 'new-instance-coordinates.csv'), 'Evolved Population (n=50)', r'_(\d+)\.pkl$')
    ]

    # Load and prepare all instance data
    new_instances = pd.concat([load_and_prepare_instance_data(path, source, pattern) for path, source, pattern in paths_and_sources])

    # Load the original data and bounds
    data = pd.read_csv('data/coordinates.csv')
    # data['Source'] = data['Row'].str.extract(r'_(\w+)$').str.title().str.replace('_', ' ')
    data['Source'] = data['Row'].str.extract(r'_(\w+)$')
    # Make source title case and remove underscores
    data['Source'] = data['Source'].str.title().str.replace('_', ' ')
    data['Population Type'] = 'Original Instances'
    # Load the bounds data
    bounds = pd.read_csv('data/bounds_prunned.csv')

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
        "Evolved Population (n=16)": "navy",
        "Evolved Population (n=24)": "green",
        "Evolved Population (n=50)": "purple"

    }
    markers = {
        "Original Instances": "o",  # Circle
        "Evolved Population (n=12)": "D",  # Diamond
        "Evolved Population (n=16)": "D",  # Diamond
        "Evolved Population (n=24)": "D",  # Diamond
        "Evolved Population (n=50)": "D"  # Diamond
    }


    plt.figure(figsize=(10, 8))

    # Filter out rows with three highest values of z_2 and row with lowest value of z_1
    # sort by z_2
    result_df = result_df.sort_values(by='z_2', ascending=False)
    result_df = result_df.iloc[3:]
    # sort by z_1
    result_df = result_df.sort_values(by='z_1', ascending=True)
    result_df = result_df.iloc[1:]

    # Plot the target point
    # for target_point in target_points:
    #     plt.scatter(target_point[0], target_point[1], marker='x', color='black', s=60)

    # Plot each group with its respective color and marker
    for population_type, group_df in result_df.groupby('Population Type'):
        # If the population type is 'Original Instances', alpha is 0.3, else 1
        alpha = 0.3 if population_type == 'Original Instances' else 0.8
        plt.scatter(group_df['z_1'], group_df['z_2'], 
                    color=colors[population_type], 
                    marker=markers[population_type], 
                    label=population_type, 
                    alpha=alpha)
        


    # Plot boundary
    boundary_points = bounds[['z_1', 'z_2']].values
    plt.gca().add_patch(Polygon(boundary_points, closed=True, fill=False, edgecolor='r', linewidth=2))


    # Set plot aesthetics
    plt.title("Evolved Instances")



    plt.xlabel("$z_1$")
    plt.ylabel("$z_2$")
    plt.legend(title="Population Type")
    plt.grid(True)

    # Write the plot to a file
    plt.savefig('evolved_instances_no_targ.png')

if __name__ == "__main__":
    main()
