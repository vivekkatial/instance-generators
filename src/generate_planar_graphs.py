# import networkx as nx
# from itertools import combinations
# from tqdm import tqdm
# import os
# import pickle

# # Create the directory for storing the graphs if it does not exist
# output_dir = "planar_graphs"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# def can_add_edge_planar(G, edge):
#     G.add_edge(*edge)
#     is_planar, _ = nx.check_planarity(G)
#     G.remove_edge(*edge)  # Remove the edge after checking
#     return is_planar

# def graph_already_seen(G, graphs_seen):
#     for H in graphs_seen:
#         if nx.is_isomorphic(G, H):
#             return True
#     return False



# N = 12
# G_base = nx.Graph()
# G_base.add_nodes_from(range(N))  # Start with N isolated nodes

# graphs_seen = [G_base.copy()]
# new_graphs = [G_base.copy()]

# # Here we attempt to add one edge at a time and check for planarity and isomorphism
# for _ in tqdm(range(N*(N-1)//2)):  # The total number of edges in a complete graph as an upper bound
#     G_temp = new_graphs.pop(0)  # Take the first graph from the list to expand
#     edges_to_consider = combinations(range(N), 2)  # All possible edges
    
#     for edge in edges_to_consider:
#         if G_temp.has_edge(*edge):
#             continue  # Skip if the edge already exists
#         if can_add_edge_planar(G_temp, edge):
#             G_temp.add_edge(*edge)
#             if not graph_already_seen(G_temp, graphs_seen):
#                 graphs_seen.append(G_temp.copy())
#                 new_graphs.append(G_temp.copy())
#             G_temp.remove_edge(*edge)  # Remove the edge to try the next one

#     if len(new_graphs) == 0:
#         break  # Exit if no new graphs can be generated

# # Writing graphs to .pkl files
# for i, G in enumerate(graphs_seen):
#     path = os.path.join(output_dir, f"planar_graph_{i}.pkl")
#     with open(path, 'wb') as file:
#         pickle.dump(G, file)

# print(f"Unique planar graphs generated and saved: {len(graphs_seen)}")


import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import random

def is_planar_and_has_room_for_more_edges(G):
    """
    Checks if the graph G is planar and if it can have more edges added
    without exceeding the maximum allowed for planarity.
    """
    return nx.check_planarity(G)[0] and G.number_of_edges() < 3 * G.number_of_nodes() - 6

def generate_planar_graph_with_min_edges(nodes, min_edges):
    G = nx.Graph()
    G.add_nodes_from(range(nodes))
    
    # Start with a minimum spanning tree to ensure initial connectivity and planarity
    edges = list(combinations(range(nodes), 2))
    random.shuffle(edges)  # Shuffle to randomize the tree structure
    
    for edge in edges:
        G.add_edge(*edge)
        if not is_planar_and_has_room_for_more_edges(G):
            G.remove_edge(*edge)  # Remove the edge if adding it violates planarity
            
        # Stop if we reach the minimum required edges
        if G.number_of_edges() >= min_edges:
            return G
    
    # If the loop ends and not enough edges have been added, try more combinations
    if G.number_of_edges() < min_edges:
        for edge in combinations(range(nodes), 2):
            if not G.has_edge(*edge):
                G.add_edge(*edge)
                if not is_planar_and_has_room_for_more_edges(G) or G.number_of_edges() >= min_edges:
                    if not is_planar_and_has_room_for_more_edges(G):
                        G.remove_edge(*edge)
                    if G.number_of_edges() >= min_edges:
                        return G
    
    # Check if the graph has the required number of edges
    if G.number_of_edges() < min_edges:
        raise ValueError(f"Failed to generate a planar graph with at least {min_edges} edges for {nodes} nodes. This graph has {G.number_of_edges()} edges.")
    
    return G

# Parameters
nodes = 12
min_edges = 23

# How do I evolve from initial population of graph with minimum 22 edge graphs

try:
    # Generate the graph
    G = generate_planar_graph_with_min_edges(nodes, min_edges)

    # Draw the graph
    print(f"Planar Graph with {G.number_of_edges()} edges and {len(G.nodes())} nodes")
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title(f"Planar Graph with {G.number_of_edges()} edges and {len(G.nodes())} nodes")
    plt.show()
except ValueError as e:
    print(e)
