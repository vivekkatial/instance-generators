import networkx as nx
import numpy as np
import pynauty as nauty

from networkx.algorithms.distance_measures import radius
from itertools import permutations


import networkx as nx
import numpy as np
import pynauty as nauty
import scipy.stats as stats

from networkx.algorithms.distance_measures import radius
from itertools import permutations

def get_graph_features(G):
    """
    Generates a list of features for the given graph

    Args:
        G (object): networkx graph object

    Returns:
        features (dict): a dictionary of the features in the given graph
    """

    features = {}

    L = nx.laplacian_matrix(G, weight="weight")
    # L  doesn't work for what we're triyng to do here (so e will not either)
    e = np.linalg.eigvals(L.A)

    features["acyclic"] = nx.is_directed_acyclic_graph(G)
    features[
        "algebraic_connectivity"
    ] = nx.linalg.algebraicconnectivity.algebraic_connectivity(G, method="lanczos")
    try:
        features["average_distance"] = nx.average_shortest_path_length(G)
    except:
        # Handle distance for dis-connected
        distances = []
        for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
            distances.append(nx.average_shortest_path_length(C))
        average_distance = np.mean(distances)
        features["average_distance"] = average_distance
    features["bipartite"] = nx.is_bipartite(G)

    # Find all cliques in the graph
    cliques = list(nx.find_cliques(G))
    features["clique_number"] = max(len(clique) for clique in cliques)
    features["connected"] = nx.algorithms.components.is_connected(G)
    features["density"] = nx.classes.function.density(G)
    if nx.algorithms.components.is_connected(G):
        features["diameter"] = nx.algorithms.distance_measures.diameter(G)
    else:
        features["diameter"] = 0
    features[
        "edge_connectivity"
    ] = nx.algorithms.connectivity.connectivity.edge_connectivity(G)
    features["eulerian"] = nx.algorithms.euler.is_eulerian(G)

    features["laplacian_largest_eigenvalue"] = max(e).real
    features["maximum_degree"] = max([G.degree[i] for i in G.nodes])
    features["minimum_degree"] = min([G.degree[i] for i in G.nodes])
    features["minimum_dominating_set"] = len(nx.algorithms.dominating.dominating_set(G))
    features[
        "number_of_components"
    ] = nx.algorithms.components.number_connected_components(G)
    features["number_of_edges"] = G.number_of_edges()
    # features['number_of_triangles'] = nx.algorithms.cluster.triangles(G)
    features["number_of_vertices"] = G.number_of_nodes()
    features["planar"] = nx.algorithms.planarity.check_planarity(G)[0]

    if nx.algorithms.components.is_connected(G):
        features["radius"] = nx.algorithms.distance_measures.radius(G)
    else:
        features["radius"] = 0

    features["regular"] = nx.algorithms.regular.is_regular(G)
    features["laplacian_second_largest_eigenvalue"] = sorted(e)[1].real
    features["ratio_of_two_largest_laplacian_eigenvaleus"] = (
        max(e).real / sorted(e)[1].real
    )
    features["smallest_eigenvalue"] = min(e).real
    features[
        "vertex_connectivity"
    ] = nx.algorithms.connectivity.connectivity.node_connectivity(G)

    # Additional features based on (https://arxiv.org/pdf/2102.05997.pdf)
    # First we need to make a Nauty graph to leverage `pynauty`
    adj_dict = {int(node): [int(neighbor) for neighbor in neighbors] for node, neighbors in G.adjacency()}
    G_pynauty = nauty.Graph(
        number_of_vertices=G.number_of_nodes(), directed=False, adjacency_dict=adj_dict
    )
    nauty_feats = nauty.autgrp(G_pynauty)

    features["number_of_cut_vertices"] = number_of_cut_vertices(G)
    features["number_of_minimal_odd_cycles"] = count_minimal_odd_cycles(G)
    features["group_size"] = calculate_group_size(G_pynauty)  # Based on PyNauty
    features["number_of_orbits"] = nauty_feats[-1]  # Based on PyNauty
    features["is_distance_regular"] = nx.is_distance_regular(G)
    features["entropy"] = get_shannon_entropy(G, adj_dict)

    return features

def get_weighted_graph_features(G):
    """
    Generates a list of weight-related features for the given connected weighted graph.

    Args:
        G (object): networkx graph object with weights

    Returns:
        features (dict): a dictionary of the weight-specific features in the given graph
    """

    if not nx.is_connected(G):
        raise ValueError("The graph must be connected to analyze weighted features.")

    features = {}

    # Check if any edge has a 'weight' attribute
    if any('weight' in data for _, _, data in G.edges(data=True)):
        weights = [data['weight'] for _, _, data in G.edges(data=True)]
    else:
        weights = [1] * G.number_of_edges()

    # Basic weight statistics
    features['mean_weight'] = float(np.mean(weights))
    features['median_weight'] = float(np.median(weights))
    features['std_dev_weight'] = float(np.std(weights))
    features['min_weight'] = float(np.min(weights))
    features['max_weight'] = float(np.max(weights))
    features['range_weight'] = features['max_weight'] - features['min_weight']
    features['skewness_weight'] = stats.skew(weights)
    features['kurtosis_weight'] = stats.kurtosis(weights)
    
    # Quantile-based features
    features['first_quartile'] = np.percentile(weights, 25)
    features['third_quartile'] = np.percentile(weights, 75)
    features['interquartile_range'] = features['third_quartile'] - features['first_quartile']

    # Extremes and variability
    features['variance_weight'] = np.var(weights)
    features['coefficient_of_variation'] = features['std_dev_weight'] / features['mean_weight'] if features['mean_weight'] != 0 else float('inf')

    # Weighted graph properties
    features['weighted_average_clustering'] = nx.average_clustering(G, weight='weight')
    features['weighted_average_shortest_path_length'] = nx.average_shortest_path_length(G, weight='weight')

    # Weighted Diameter and Radius
    features['weighted_diameter'] = nx.diameter(G)
    features['weighted_radius'] = nx.radius(G)

    # Weighted Degree
    weighted_degrees = {node: sum(data['weight'] if 'weight' in data else 1 for _, data in G[node].items()) for node in G.nodes()}
    features['maximum_weighted_degree'] = max(weighted_degrees.values())
    features['minimum_weighted_degree'] = min(weighted_degrees.values())
    # If anything is nan, replace with 0
    for key, value in features.items():
        if np.isnan(value):
            features[key] = 0
    
    return features

def is_subcycle(small_cycle, big_cycle):
    """
    Checks if small_cycle is a subcycle of big_cycle.
    """
    return all(node in big_cycle for node in small_cycle)

def count_minimal_odd_cycles(graph):
    """
    Counts the number of minimal odd cycles in a graph. A
    minimal odd cycle is an odd-length cycle that does not contain any other odd cycle within it.

    Parameters:
    graph (networkx.Graph): The graph to be analyzed.

    Returns:
    int: The number of minimal odd cycles in the graph.
    """
    # Finding all cycles in the graph
    cycles = nx.cycle_basis(graph)

    # Filtering odd cycles
    odd_cycles = [cycle for cycle in cycles if len(cycle) % 2 != 0]

    # Identifying minimal odd cycles
    minimal_odd_cycles = []
    for cycle in odd_cycles:
        if not any(
            is_subcycle(possible_subcycle, cycle)
            for possible_subcycle in odd_cycles
            if possible_subcycle != cycle
        ):
            minimal_odd_cycles.append(cycle)

    return len(minimal_odd_cycles)

def number_of_cut_vertices(G):
    """
    Calculate the number of cut vertices in the graph G.

    Parameters:
    G (networkx.Graph): A networkx graph.

    Returns:
    int: The number of cut vertices in G.
    """
    return len(list(nx.articulation_points(G)))

def calculate_group_size(G):
    """
    Calculate the size of the automorphism group of graph G. Based on https://users.cecs.anu.edu.au/~bdm/nauty/nug26.pdf

    Within rounding error, the order of the automorphism group is equal to grpsize1 * 10^(grpsize2)

    Parameters:
    G (pynauty.graph.Graph): A pyNauty graph.

    Returns:
    int: The size of the automorphism group of G.
    """
    grpsize1 = nauty.autgrp(G)[1]
    grpsize2 = nauty.autgrp(G)[2]

    group_size = grpsize1 * (10**grpsize2)

    return group_size

from collections import defaultdict, Counter

def get_shannon_entropy(G, adjacency_dict):
    """ Calculate the Shannon entropy of the graph G using the implementation in
    https://arxiv.org/pdf/2012.04713

    Parameters:
    G (networkx.Graph): A networkx graph.
    """

    g = nauty.Graph(number_of_vertices=G.number_of_nodes(), directed=nx.is_directed(G),
                adjacency_dict = adjacency_dict)
    aut = nauty.autgrp(g)
    S = 0
    for orbit, orbit_size in Counter(aut[3]).items():
        S += ((orbit_size * np.log(orbit_size)) / G.number_of_nodes())
    return S



def build_feature_df(G, source):

    graph_features = get_graph_features(G)
    weighted_features = get_weighted_graph_features(G)

    # Map the features to the ISA metadata csv
    mapped_features = {
        'Source': source,
        'feature_density': graph_features['density'],
        'feature_radius': graph_features['radius'],
        'feature_minimum_degree': graph_features['minimum_degree'],
        'feature_algebraic_connectivity': graph_features['algebraic_connectivity'],
        # 'feature_connected': graph_features['connected'],
        'feature_number_of_cut_vertices': graph_features['number_of_cut_vertices'],
        'feature_minimum_dominating_set': graph_features['minimum_dominating_set'],
        'feature_diameter': graph_features['diameter'],
        'feature_laplacian_second_largest_eigenvalue': graph_features['laplacian_second_largest_eigenvalue'],
        # 'feature_number_of_components': graph_features['number_of_components'],
        'feature_smallest_eigenvalue': graph_features['smallest_eigenvalue'],
        'feature_regular': graph_features['regular'],
        'feature_planar': graph_features['planar'],
        'feature_bipartite': graph_features['bipartite'],
        'feature_clique_number': graph_features['clique_number'],
        'feature_eulerian': graph_features['eulerian'],
        'feature_average_distance': graph_features['average_distance'],
        'feature_edge_connectivity': graph_features['edge_connectivity'],
        'feature_maximum_degree': graph_features['maximum_degree'],
        'feature_vertex_connectivity': graph_features['vertex_connectivity'],
        'feature_laplacian_largest_eigenvalue': graph_features['laplacian_largest_eigenvalue'],
        'feature_number_of_orbits': graph_features['number_of_orbits'],
        'feature_ratio_of_two_largest_laplacian_eigenvaleus': graph_features['ratio_of_two_largest_laplacian_eigenvaleus'],
        'feature_group_size': graph_features['group_size'],
        'feature_number_of_edges': graph_features['number_of_edges'],
        'feature_number_of_minimal_odd_cycles': graph_features['number_of_minimal_odd_cycles'],
        'feature_weighted_average_clustering': weighted_features['weighted_average_clustering'],
        'feature_weighted_average_shortest_path_length': weighted_features['weighted_average_shortest_path_length'],
        'feature_weighted_diameter': weighted_features['weighted_diameter'],
        'feature_weighted_radius': weighted_features['weighted_radius'],
        'feature_coefficient_of_variation': weighted_features['coefficient_of_variation'],
        'feature_entropy': graph_features['entropy'],
        'feature_first_quartile': weighted_features['first_quartile'],
        'feature_interquartile_range': weighted_features['interquartile_range'],
        'feature_is_distance_regular': graph_features['is_distance_regular'],
        'feature_kurtosis_weight': weighted_features['kurtosis_weight'],
        'feature_maximum_weighted_degree': weighted_features['maximum_weighted_degree'],
        'feature_mean_weight': weighted_features['mean_weight'],
        'feature_median_weight': weighted_features['median_weight'],
        'feature_minimum_weighted_degree': weighted_features['minimum_weighted_degree'],
        'feature_skewness_weight': weighted_features['skewness_weight'],
        'feature_std_dev_weight': weighted_features['std_dev_weight'],
        'feature_third_quartile': weighted_features['third_quartile'],
        'feature_variance_weight': weighted_features['variance_weight'],
        'algo_fixed_angles_constant': 0,
        'algo_fourier': 0,
        'algo_interp': 0,
        'algo_qibpi': 0,
        'algo_random': 0,
        'algo_three_regular': 0,
        'algo_tqa': 0,
    }

    return mapped_features
