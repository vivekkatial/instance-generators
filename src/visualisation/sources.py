import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import json
import networkx as nx
import pickle
import os

from matplotlib.patches import Polygon
from src.preprocess import get_z1_z2_projection

print('=========================================================================')
print('-> Plotting Instances.')
print('=========================================================================')
plot_bounds = False
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

# Print head of the data
print(data.head())


# # Set the aesthetic style of the plots
sns.set_style("whitegrid")
sns.set_context("paper")

# # Create a scatter plot colored by the extracted source

scatter = sns.scatterplot(
    data=data, x='z_1', y='z_2', palette='deep', alpha=1, s=10, edgecolor='black', color='grey'
)

# # Extract the boundary points and plot the polygon
if plot_bounds:
    bounds = pd.read_csv(f"{experiment}/bounds.csv")
    boundary_points = bounds[['z_1', 'z_2']].values
    polygon = Polygon(boundary_points, closed=True, fill=False, edgecolor='r', linewidth=2)
    plt.gca().add_patch(polygon)

    all_x = pd.concat([data['z_1'], pd.Series(boundary_points[:, 0])])
    all_y = pd.concat([data['z_2'], pd.Series(boundary_points[:, 1])])
else:
    all_x = data['z_1']
    all_y = data['z_2']

# Dynamically adjust the axes limits considering all data, target points, and boundary points
plt.xlim([all_x.min() - 1, all_x.max() + 1])
plt.ylim([all_y.min() - 1, all_y.max() + 1])

# Take legend off the plot but ensure frame is visible
# Ensure we can see the legend
plt.tight_layout()

TARGET= [1.5, 0.57]
# Read in and plot the graph
f=f"{experiment}/target-point-graphs/target_point_{TARGET[0]}_{TARGET[1]}_n_12/best_graph_gen_323.graphml"
G = nx.read_graphml(f)

# Project the graph onto the z1 and z2 axes
projection = get_z1_z2_projection(G, experiment=experiment)

# Add the projection to the plot as a yellow star
plt.scatter(projection[0], projection[1], marker='*', color='red', s=40)
# Add the target point to the plot as a blue star
plt.scatter(TARGET[0], TARGET[1], marker='*', color='blue', s=100)



# Show the new plot
plt.show()

