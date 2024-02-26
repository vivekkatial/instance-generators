import networkx as nx
import matplotlib.pyplot as plt
import argparse
import json
from graph_features import get_graph_features

# Load graph based on command args
parser = argparse.ArgumentParser(description='Visualise graph')
parser.add_argument(
    '--graph',
    type=str,
    help='Path to the graph file',
    default='runs-to-keep/best_graph_50.pkl',
)

args = parser.parse_args()
G = nx.read_gpickle(args.graph)

features = get_graph_features(G)
print(json.dumps(features, indent=4))
# Plot graph, if planar, do planar layout else do spring layout
if nx.algorithms.planarity.check_planarity(G)[0]:
    pos = nx.planar_layout(G)
else:
    pos = nx.spring_layout(G)

plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, font_weight='bold')
plt.show()

# Save graph features to file
with open('graph_features.json', 'w') as f:
    json.dump(features, f, indent=4)

