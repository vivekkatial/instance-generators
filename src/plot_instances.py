import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Custom imports from the project
# from preprocess import generate_new_instance, project_and_map

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
scatter = sns.scatterplot(data=data, x='z_1', y='z_2', hue='Source', palette='deep', alpha=0.5, s=50)

# Extract the boundary points and plot the polygon
boundary_points = bounds[['z_1', 'z_2']].values
polygon = Polygon(boundary_points, closed=True, fill=False, edgecolor='r', linewidth=2)
plt.gca().add_patch(polygon)

# New projections
new_instances_path = 'data/new-instance-coordinates.csv'  # Adjust the path if necessary
new_inst_df = pd.read_csv(new_instances_path)

# Plot each point and annotate it
for index, row in new_inst_df.iterrows():
    plt.scatter(row['z_1'], row['z_2'], marker='*', color='blue', s=50, alpha=0.5)
    plt.text(row['z_1'], row['z_2'], row['Source'], color='black', fontsize=9, ha='right', va='bottom')

# plot a target point and mark it
target_point = [2.5, 2.5]
plt.scatter(target_point[0], target_point[1], marker='x', color='black', s=100)

# Plot over range of target points
target_points = [[2.5, 2.5], [3.75, 0]]
for point in target_points:
    plt.scatter(point[0], point[1], marker='x', color='black', s=100)


# Extract all x and y coordinates including data, target points, and boundary points
all_x = pd.concat([data['z_1'], pd.Series(boundary_points[:, 0])])
all_y = pd.concat([data['z_2'], pd.Series(boundary_points[:, 1])])

# Dynamically adjust the axes limits considering all data, target points, and boundary points
plt.xlim([all_x.min() - 1, all_x.max() + 1])
plt.ylim([all_y.min() - 1, all_y.max() + 1])

# Adding grid points with x marks for every whole number coordinate
x_min, x_max = all_x.min() - 1, all_x.max() + 1
y_min, y_max = all_y.min() - 1, all_y.max() + 1
for x in range(int(x_min), int(x_max) + 1):
    for y in range(int(y_min), int(y_max) + 1):
        plt.scatter(x, y, marker='x', color='k', s=0.5)  # 'k' denotes black color

# Enhance the plot for readability and aesthetics
scatter.set_title('Scatter Plot of Instances with Boundary by Source')
scatter.set_xlabel('$Z_1$')
scatter.set_ylabel('$Z_2$')
plt.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left')

# Tight layout for saving without cutting off labels
plt.tight_layout()

# Save the figure
plt.savefig('scatter_plot_with_boundary_by_source.pdf')
# Show the plot
plt.show()
