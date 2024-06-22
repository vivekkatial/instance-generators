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
    new_instances_path = os.path.join(experiment, 'new-instance-coordinates.csv')

    # paths_and_sources = [
    #     (os.path.join(experiment, 'best_graphs_12', 'new-instance-coordinates.csv'), 'Evolved Population (n=12)', r'_(\d+)\.graphml$'),
    #     (os.path.join(experiment, 'best_graphs_14', 'new-instance-coordinates.csv'), 'Evolved Population (n=14)', r'_(\d+)\.graphml$'),
    #     # (os.path.join('best_graphs_24', 'new-instance-coordinates.csv'), 'Evolved Population (n=24)', r'_(\d+)\.graphml$'),
    #     # (os.path.join('best_graphs_50', 'new-instance-coordinates.csv'), 'Evolved Population (n=50)', r'_(\d+)\.graphml$')
    # ]

    # # Load and prepare all instance data
    # new_instances = pd.concat([load_and_prepare_instance_data(path, source, pattern) for path, source, pattern in paths_and_sources])

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

    # Check if the new instances file exists
    if not os.path.exists(new_instances_path):
        print(f'No new instances found at {new_instances_path}. Plotting source dist...')
        # Create a scatter plot colored by the extracted source
        scatter = sns.scatterplot(
            data=data, x='z_1', y='z_2', hue='Source', palette='deep', alpha=0.5, s=50
        )

        for target_point in target_points:
            plt.scatter(target_point[0], target_point[1], marker='x', color='black', s=60)


        # Extract the boundary points and plot the polygon
        boundary_points = bounds[['z_1', 'z_2']].values
        polygon = Polygon(boundary_points, closed=True, fill=False, edgecolor='r', linewidth=2)
        plt.gca().add_patch(polygon)
        # Save the plot
        # plt.show()
        plt.savefig('original_scatter.png')
    else:
        result_df = pd.concat([new_instances, data], ignore_index=True)
    
    # Define colors and markers
    colors = {
        "Original Instances": "grey",
        "Evolved Population (n=12)": "brown",
        "Evolved Population (n=14)": "navy",
        "Evolved Population (n=24)": "green",
        "Evolved Population (n=50)": "purple"

    }
    markers = {
        "Original Instances": "o",  # Circle
        "Evolved Population (n=12)": "D",  # Diamond
        "Evolved Population (n=14)": "D",  # Diamond
        "Evolved Population (n=24)": "D",  # Diamond
        "Evolved Population (n=50)": "D"  # Diamond
    }


    plt.figure(figsize=(10, 8))

    # Plot the target point

    # # Plot each group with its respective color and marker
    # for population_type, group_df in result_df.groupby('Population Type'):
    #     # If the population type is 'Original Instances', alpha is 0.3, else 1
    #     alpha = 0.3 if population_type == 'Original Instances' else 0.8
    #     plt.scatter(group_df['z_1'], group_df['z_2'], 
    #                 color=colors[population_type], 
    #                 marker=markers[population_type], 
    #                 label=population_type, 
    #                 alpha=alpha)
        


    # Plot boundary
    boundary_points = bounds[['z_1', 'z_2']].values
    plt.gca().add_patch(Polygon(boundary_points, closed=True, fill=False, edgecolor='r', linewidth=2))


    # Set plot aesthetics
    # plt.title("Evolved Instances")



    plt.xlabel("$z_1$")
    plt.ylabel("$z_2$")
    plt.legend(title="Population Type")
    plt.grid(True)

    # Write the plot to a file
    plt.savefig('new_scatter.png')

if __name__ == "__main__":
    main()
