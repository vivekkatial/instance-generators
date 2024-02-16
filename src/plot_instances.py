import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Custom imports from the project
from preprocess import generate_new_instance, project_and_map

# Assuming 'data' is already loaded and contains a 'Row' column from which to extract the source
# Extract the source from the 'Row' column
data = pd.read_csv('data/coordinates.csv')
data['Source'] = data['Row'].str.extract(r'_(\w+)$')

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



# Define and plot target points
target_points = [(4.2, 0)]  # Add more points as needed
for tp in target_points:
    plt.scatter(tp[0], tp[1], marker='*', color='blue', s=100)  # 's' adjusts the size of the star

# Generate a new graph instance and project it
# import pdb; pdb.set_trace()
features = generate_new_instance()
projected_instances = project_and_map(features, 'data/projection_matrix.csv')

# Add the projected instance to the plot with a triangle marker
for i in projected_instances:
    projected_instance = i[0], i[1]
    plt.scatter(projected_instance[0], projected_instance[1], marker='^', color='green', s=100)  # Triangle marker
# Add the projected instance to your dataset for plotting
new_instance_df = pd.DataFrame([{'z_1': projected_instance[0], 'z_2': projected_instance[1], 'Source': 'Generated'}])
data = pd.concat([data, new_instance_df], ignore_index=True)


# Dynamically adjust the axes limits considering all data, including the new instance
all_x = pd.concat([data['z_1'], bounds['z_1'], pd.Series(projected_instance[0])])
all_y = pd.concat([data['z_2'], bounds['z_2'], pd.Series(projected_instance[1])])
plt.xlim([all_x.min() - 1, all_x.max() + 1])
plt.ylim([all_y.min() - 1, all_y.max() + 1])

# Adding grid points with x marks for every whole number coordinate
x_min, x_max = all_x.min() - 1, all_x.max() + 1
y_min, y_max = all_y.min() - 1, all_y.max() + 1
for x in range(int(x_min), int(x_max) + 1):
    for y in range(int(y_min), int(y_max) + 1):
        plt.scatter(x, y, marker='x', color='k', s=1)  # 'k' denotes black color
        
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
