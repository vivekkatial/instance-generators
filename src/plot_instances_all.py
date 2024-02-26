import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import json
import networkx as nx
import os

from targets import target_points
from matplotlib.patches import Polygon
from graph_features import get_graph_features

print('=========================================================================')
print('-> Plotting Instances.')
print('=========================================================================')

load_path = "best_graphs/"
new_instances_path = os.path.join(load_path, 'new-instance-coordinates.csv')

# Custom imports from the project
# Assuming 'data' is already loaded and contains a 'Row' column from which to extract the source
data = pd.read_csv('data/coordinates.csv')
data['Source'] = data['Row'].str.extract(r'_(\w+)$')
# Make source title case and remove underscores
data['Source'] = data['Source'].str.title().str.replace('_', ' ')

# Load the bounds data
bounds = pd.read_csv('data/bounds.csv')

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


for target_point in target_points:
    plt.scatter(target_point[0], target_point[1], marker='x', color='black', s=60)

# New projections
new_inst_df = pd.read_csv(new_instances_path)
# Extract generation number based on the file name `best_graph_{gen}.pkl`
new_inst_df['Generation'] = (
    new_inst_df['Source'].str.extract(r'_(\d+)\.pkl$').astype(int)
)


# Filter new instances to include the final population (source contains `final_population`)
final_population = new_inst_df[new_inst_df['Source'].str.contains('final_population')]
# Plot each point and annotate it
for index, row in final_population.iterrows():
    plt.scatter(row['z_1'], row['z_2'], marker='D', color='brown', s=20, alpha=1)




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
