import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import json
import networkx as nx
import os

from matplotlib.patches import Polygon
from src.preprocess import get_z1_z2_projection

print('=========================================================================')
print('-> Plotting Instances.')
print('=========================================================================')

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot instances')
parser.add_argument(
    "--experiment",
    type=str,
    nargs=1,
    default="",
    help="The expeirment.",
)
args = parser.parse_args()
experiment = args.experiment[0]

# Check if the experiment directory exists
if not os.path.exists(f"{experiment}"):
    print(f"Experiment {experiment} does not exist.")
    exit()

# Load the data
data = pd.read_csv(f"{experiment}/coordinates.csv")
data['Source'] = data['Row'].str.extract(r'_(\w+)$')
data['Source'] = data['Source'].str.title().str.replace('_', ' ')

# Print head of the data
print(data.head())

bounds = pd.read_csv(f"{experiment}/bounds_prunned.csv")

# # Set the aesthetic style of the plots
sns.set_style("whitegrid")
sns.set_context("paper")

# # Create a scatter plot colored by the extracted source

scatter = sns.scatterplot(
    data=data, x='z_1', y='z_2', hue='Source', palette='deep', alpha=1, s=25, edgecolor='black'
)

# # Extract the boundary points and plot the polygon
boundary_points = bounds[['z_1', 'z_2']].values
polygon = Polygon(boundary_points, closed=True, fill=False, edgecolor='r', linewidth=2)
plt.gca().add_patch(polygon)

all_x = pd.concat([data['z_1'], pd.Series(boundary_points[:, 0])])
all_y = pd.concat([data['z_2'], pd.Series(boundary_points[:, 1])])

# Dynamically adjust the axes limits considering all data, target points, and boundary points
plt.xlim([all_x.min() - 1, all_x.max() + 1])
plt.ylim([all_y.min() - 1, all_y.max() + 1])

# Take legend off the plot but ensure frame is visible
plt.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left')
# Ensure we can see the legend
plt.tight_layout()

# List of target points
target_points = [
    (-3.0, 2.2),
    (-3.15, 2.6),
    (-3.3, 3.0),
    (-3.45, 3.4),
    (-3.6, 3.8),
    (-3.75, 4.2),
    (-3.9, 4.6),
    (-4.05, 5.0),
    (-4.2, 5.4),
    (-4.0, 5.2),
    (-3.8, 5.0),
    (-3.6, 4.8),
    (-3.4, 4.6),
    (-4.0, 5.4),
    (-3.5, 5.375),
    (-3.0, 5.35),
    (-2.5, 5.325),
    (-2.0, 5.3),
    (-1.5, 5.275),
    (-1.0, 5.25),
    (-0.5, 5.225),
    (0, 5.2),
    (-1.8, -1),
    (-1.1, -1.7),
    (-1.8, -1.5),
    (-1.3, -1.5),
    (-1.5, -2.0),
    (-1.0, -2.0 ),
    (-0.5, -2.0),
    (-1.4, -2.73),
    (-1.0, -2.73),
    (-0.5, -2.73),
    (0.0, -2.73),
    (0.5,  -2.73),
    (-1, -3.35),
    (-0.5, -3.35),
    (0.0, -3.35),
    (0.5, -3.35),
    (1.0, -3.35),
    (-1.0, -4.0),
    (-0.5, -4.0),
    (0.0, -4.0),
    (0.5, -4.0),
    (1.0, -4.0),
    (1.5, -4.0),
    (2.0, -4.0),
    (0, -4.5),
    (0.5, -4.5),
    (1.0, -4.5),
    (1.5, -4.5),
    (2.0, -4.5),
    (2.5, -4.5),
    (3.0, -5),
]

bite_target_points = [
    (-2.5, 4),
    (-2.5, 4.5),
    (-2.25, 2.27),
    (-2.25, 3.55),
    (-2, 2),
    (-2, 3),
    (-1.5, 3.23),
    (-1.11, 4),
    (0, 3.35),
    (0, 4),
    (0.28, 2.8),
    (0.375, 4),
    (0.4, 2.3),
    (0.59, 1.5),
    (0.75, 3),
    (0.75, 3),
    (0.75, 4)
]

# Plot each target point and annotate it
for target_point in target_points:
    plt.scatter(target_point[0], target_point[1], marker='x', color='black', s=100)

for target_point in bite_target_points:
    plt.scatter(target_point[0], target_point[1], marker='x', color='blue', s=100)

TARGET= [2.5, 4]
# Read in and plot the graph
f=f"{experiment}/target-point-graphs/target_point_2.5_4.0_n_50/best_graph_gen_737.pkl"
# Read in the graph pkl file
G = nx.read_gpickle(f)
# Project the graph onto the z1 and z2 axes
projection = get_z1_z2_projection(G, experiment=experiment)

# Add the projection to the plot as a yellow star
plt.scatter(projection[0], projection[1], marker='*', color='yellow', s=100)




# Show the new plot
plt.show()

