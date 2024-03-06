import matplotlib.pyplot as plt
import numpy as np

def is_point_inside_polygon(x, y, polygon):
    num = len(polygon)
    j = num - 1
    c = False
    for i in range(num):
        if ((polygon[i][1] > y) != (polygon[j][1] > y)) and \
           (x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) / (polygon[j][1] - polygon[i][1]) + polygon[i][0]):
            c = not c
        j = i
    return c

def generate_grid_points_within_polygon(polygon, grid_spacing):
    min_x = min(polygon, key=lambda point: point[0])[0]
    max_x = max(polygon, key=lambda point: point[0])[0]
    min_y = min(polygon, key=lambda point: point[1])[1]
    max_y = max(polygon, key=lambda point: point[1])[1]
    
    x_range = np.arange(min_x, max_x, grid_spacing)
    y_range = np.arange(min_y, max_y, grid_spacing)
    
    points = []
    for x in x_range:
        for y in y_range:
            if is_point_inside_polygon(x, y, polygon):
                points.append((x, y))
    
    return points

# Define the polygon using the provided bounds
polygon = [
    (-1.57432244995894, -8.47487734409522),
    (-0.285883063945643, -8.60353157166864),
    (3.5299736236439, -8.88436100853421),
    (5.98283494197395, -9.01949542518735),
    (8.6336900966105, -5.46001961121988),
    (9.69858047670594, -3.29518815298139),
    (9.58143029376066, 0.625374964708605),
    (0.788073663922, 8.95165789465186),
    (-0.5003657220913, 9.08031212222527),
    (-4.31622240968084, 9.36114155909084),
    (-6.7690837280109, 9.49627597574398),
    (-9.41993888264744, 5.93680016177652),
    (-10.4848292627429, 3.77196870353802),
    (-10.3676790797976, -0.14859441415197),
    (-1.57432244995894, -8.47487734409522) # Closing the polygon
]

# Generate grid points within the polygon
grid_spacing = 1 
grid_points = generate_grid_points_within_polygon(polygon, grid_spacing)

# Plot for visualization
x_poly, y_poly = zip(*polygon)
plt.fill(x_poly, y_poly, edgecolor='r', fill=False)
x_points, y_points = zip(*grid_points)
plt.scatter(x_points, y_points, color='blue', s=10) # s is the size of points
plt.show()
