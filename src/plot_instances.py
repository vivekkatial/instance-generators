import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import json
import networkx as nx
import os

from matplotlib.patches import Polygon
from graph_features import get_graph_features

print('=========================================================================')
print('-> Plotting Instances.')
print('=========================================================================')

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot instances')
parser.add_argument(
    "--target_point",
    type=float,
    nargs=2,
    default=[2.5, 2.5],
    help="The target point for the fitness function.",
)

parser.add_argument(
    "--experiment",
    type=str,
    nargs=1,
    default="INFORMS-Revision-12-node-network",
    help="The experiment.",
)


args = parser.parse_args()
target_point = args.target_point
experiment = args.experiment[0]
experiment = "INFORMS-Revision-12-node-network"

# Load new instances based on target point
load_path = os.path.join(experiment,"target-point-graphs", f"target_point_{target_point[0]}_{target_point[1]}_n_12")

new_instances_path = os.path.join(load_path, 'new-instance-coordinates.csv')

# Custom imports from the project
# Assuming 'data' is already loaded and contains a 'Row' column from which to extract the source
d_coords = pd.read_csv(f'{experiment}/coordinates.csv')
d_source = pd.read_csv(f'{experiment}/metadata.csv')
# Select only the `Instances` and `Source` columns only and then join the Source column to the coordinates (with Row = Instances)
data = d_coords[['Row', 'z_1', 'z_2']].merge(d_source[['Instances', 'Source']], left_on='Row', right_on='Instances')
# Dont keep the `Instances` column
data.drop(columns='Instances', inplace=True)

# Load the bounds data
bounds = pd.read_csv(f'{experiment}/bounds_prunned-0.9.csv')

# Set the aesthetic style of the plots
sns.set_style("whitegrid")
sns.set_context("paper")

# Create a scatter plot colored by the extracted source
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    data=data, x='z_1', y='z_2', hue='Source', palette='deep', alpha=0.5, s=50
)

# Extract the boundary points and plot the polygon
boundary_points = bounds[['z_1', 'z_2']].values
polygon = Polygon(boundary_points, closed=True, fill=False, edgecolor='r', linewidth=2)
plt.gca().add_patch(polygon)

# plot a target point and mark it with an 'x'
plt.scatter(target_point[0], target_point[1], marker='x', color='black', s=100)

# New projections
# Check if the new instances file exists
if not os.path.exists(new_instances_path):
    print(f'No new instances found at {new_instances_path}. Exiting...')

else:

    new_inst_df = pd.read_csv(new_instances_path)
    # Extract generation number based on the file name `best_graph_{gen}.graphml`
    new_inst_df['Generation'] = (
        new_inst_df['Source'].str.extract(r'_(\d+)\.graphml$').astype(int)
    )

# Filter new instances to include the final population (source contains `final_population`)
final_population = new_inst_df[new_inst_df['Source'].str.contains('final_population')]
# Plot each point and annotate it
for index, row in final_population.iterrows():
    plt.scatter(row['z_1'], row['z_2'], marker='D', color='brown', s=20, alpha=1)

# Plot each point and annotate it
for index, row in new_inst_df.iterrows():
    # If generation is 1 and source contains `best_graph`, plot as a circle
    if row['Generation'] == 0 and 'best_graph' in row['Source']:
        plt.scatter(
            row['z_1'], row['z_2'], marker='o', color='red', s=200, edgecolor='black'
        )
        plt.text(
            row['z_1'],
            row['z_2'],
            row['Source'],
            color='black',
            fontsize=9,
            ha='right',
            va='bottom',
        )
    # If maxmimum generation, plot as a star
    elif (
        row['Generation'] == new_inst_df['Generation'].max()
        and 'best_graph' in row['Source']
    ):
        plt.scatter(
            row['z_1'],
            row['z_2'],
            marker='*',
            color='gold',
            s=400,
            alpha=1,
            edgecolor='black',
        )
        plt.text(
            row['z_1'],
            row['z_2'],
            row['Source'],
            color='black',
            fontsize=9,
            ha='right',
            va='bottom',
        )


# Extract all x and y coordinates including data, target points, and boundary points
all_x = pd.concat([data['z_1'], pd.Series(boundary_points[:, 0])])
all_y = pd.concat([data['z_2'], pd.Series(boundary_points[:, 1])])

# Dynamically adjust the axes limits considering all data, target points, and boundary points
plt.xlim([all_x.min() - 1, all_x.max() + 1])
plt.ylim([all_y.min() - 1, all_y.max() + 1])

# Enhance the plot for readability and aesthetics
scatter.set_title('Scatter Plot of Instances with Boundary by Source')
scatter.set_xlabel('$Z_1$')
scatter.set_ylabel('$Z_2$')
plt.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left')

# Tight layout for saving without cutting off labels
plt.tight_layout()

# Save the figure to target point folder
plt.savefig(f"{load_path}/scatter_plot.png", bbox_inches="tight")

print('=========================================================================')
print('-> Plotting Network for Evolved Instance.')
print('=========================================================================')

# Load the best graph from the final generation
G = nx.read_graphml(f"{load_path}/best_graph_gen_{new_inst_df['Generation'].max()}.graphml")

features = get_graph_features(G)
print(json.dumps(features, indent=4))
# Plot graph, if planar, do planar layout else do spring layout
if nx.algorithms.planarity.check_planarity(G)[0]:
    pos = nx.planar_layout(G)
else:
    pos = nx.spring_layout(G)

plt.clf()
plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, font_weight='bold')
# Save the figure to target point folder
plt.savefig(f"{load_path}/evolved_network_plot.png", bbox_inches="tight")

# Save graph features to file
with open(f"{load_path}/graph_features.json", 'w') as f:
    json.dump(features, f, indent=4)



print('=========================================================================')
print(f'-> Completed GA Evolution -- check the results in the {load_path} folder.')
print('=========================================================================')
