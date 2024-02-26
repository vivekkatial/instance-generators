import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import random


def is_planar_and_has_room_for_more_edges(G):
    """
    Checks if the graph G is planar and if it can have more edges added
    without exceeding the maximum allowed for planarity.
    """
    return (
        nx.check_planarity(G)[0] and G.number_of_edges() < 3 * G.number_of_nodes() - 6
    )


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
                if (
                    not is_planar_and_has_room_for_more_edges(G)
                    or G.number_of_edges() >= min_edges
                ):
                    if not is_planar_and_has_room_for_more_edges(G):
                        G.remove_edge(*edge)
                    if G.number_of_edges() >= min_edges:
                        return G

    # Check if the graph has the required number of edges from the ISA
    if G.number_of_edges() <= 10:
        raise ValueError(
            f"Failed to generate a planar graph with at least {min_edges} edges for {nodes} nodes. This graph has {G.number_of_edges()} edges."
        )

    return G
