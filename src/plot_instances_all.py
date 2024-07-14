import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Polygon
from src.data.targets_evolved import target_points

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

    experiment = 'INFORMS-Revision-12-node-network'
    # new_instances_path = os.path.join(experiment, 'new-instance-coordinates.csv')

    paths_and_sources = [
        (os.path.join(experiment, 'best_graphs_12', 'new-instance-coordinates.csv'), 'Evolved Population (n=12)', r'_(\d+)\.graphml$'),
        (os.path.join(experiment, 'best_graphs_20', 'new-instance-coordinates.csv'), 'Evolved Population (n=20)', r'_(\d+)\.graphml$'),
        # (os.path.join('best_graphs_24', 'new-instance-coordinates.csv'), 'Evolved Population (n=24)', r'_(\d+)\.graphml$'),
        # (os.path.join('best_graphs_50', 'new-instance-coordinates.csv'), 'Evolved Population (n=50)', r'_(\d+)\.graphml$')
    ]

    # # Load and prepare all instance data
    new_instances = pd.concat([load_and_prepare_instance_data(path, source, pattern) for path, source, pattern in paths_and_sources])

    # Load the original data and bounds
    # Assuming 'data' is already loaded and contains a 'Row' column from which to extract the source
    d_coords = pd.read_csv(f'{experiment}/coordinates.csv')
    d_source = pd.read_csv(f'{experiment}/metadata.csv')
    # Select only the `Instances` and `Source` columns only and then join the Source column to the coordinates (with Row = Instances)
    data = d_coords[['Row', 'z_1', 'z_2']].merge(d_source[['Instances', 'Source']], left_on='Row', right_on='Instances')
    # Dont keep the `Instances` column
    data.drop(columns='Instances', inplace=True)
    # Make source title case and remove underscores
    data['Population Type'] = 'Original Instances'
    # Load the bounds data
    bounds = pd.read_csv(f'{experiment}/bounds_prunned-0.85.csv')

    # Set plot aesthetics
    sns.set(style="whitegrid", context="paper")
    plt.figure(figsize=(10, 10))
    plt.rcParams['figure.dpi'] = 300

    # Plot boundary
    boundary_points = bounds[['z_1', 'z_2']].values
    # plt.gca().add_patch(Polygon(boundary_points, closed=True, fill=False, edgecolor='r', linewidth=2))

    result_df = pd.concat([new_instances, data], ignore_index=True)
    
    # Define colors and markers
    colors = {
        "Original Instances": "grey",
        "Evolved Population (n=12)": "brown",
        "Evolved Population (n=20)": "navy",
        "Evolved Population (n=24)": "green",
        "Evolved Population (n=50)": "purple"

    }
    markers = {
        "Original Instances": "o",  # Circle
        "Evolved Population (n=12)": "D",  # Diamond
        "Evolved Population (n=20)": "D",  # Diamond
        "Evolved Population (n=24)": "D",  # Diamond
        "Evolved Population (n=50)": "D"  # Diamond
    }


    plt.figure(figsize=(10, 8))

    # Plot the target point
    for target_point in target_points:
        plt.scatter(target_point[0], target_point[1], color='black', marker='x', s=10)

    # # Plot each group with its respective color and marker
    # Plot Original Instances separately to ensure they are plotted first
    original_instances = result_df[result_df['Population Type'] == 'Original Instances']
    plt.scatter(original_instances['z_1'], original_instances['z_2'], 
                color=colors['Original Instances'], 
                marker=markers['Original Instances'], 
                label='Original Instances', 
                alpha=0.3)

    # Plot the rest of the groups
    for population_type, group_df in result_df.groupby('Population Type'):
        # Skip plotting Original Instances again
        if population_type == 'Original Instances':
            continue
        alpha = 0.8
        plt.scatter(group_df['z_1'], group_df['z_2'], 
                    color=colors[population_type], 
                    marker=markers[population_type], 
                    label=population_type, 
                    alpha=alpha)


    # Plot boundary
    # boundary_points = bounds[['z_1', 'z_2']].values
    # plt.gca().add_patch(Polygon(boundary_points, closed=True, fill=False, edgecolor='r', linewidth=2))


    # Set plot aesthetics
    plt.title("Evolved Instances")



    plt.xlabel("$z_1$")
    plt.ylabel("$z_2$")
    plt.legend(title="Population Type")
    plt.grid(True)

    # Write the plot to a file
    plt.savefig('evolved_instances_all.png')

if __name__ == "__main__":
    main()
