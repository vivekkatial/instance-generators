import networkx as nx
import random
import numpy as np
import os
import pickle

from preprocess_manual import get_z1_z2_projection

# Parameters
population_size = 10
mutation_rate = 0.2
crossover_rate = 0.5
number_of_nodes = 12
generations = 100
save_path = "best_graphs"  # Directory to save best graphs

if not os.path.exists(save_path):
    os.makedirs(save_path)

def calculate_fitness(graph):
    """Euclidean Distance of Z_1, Z_2 projection to a target point."""
    target_point = np.array([-5, 5])
    coordinate_matrix_of_graph = get_z1_z2_projection(graph)
    return -np.linalg.norm(coordinate_matrix_of_graph - target_point)

def initialize_population(size, number_of_nodes):
    # Start off with Power Law Graphs
    return [nx.random_geometric_graph(number_of_nodes, 0.5) for _ in range(size)]

def select_parent(population, fitnesses):
    tournament = random.sample(list(zip(population, fitnesses)), 3)
    return max(tournament, key=lambda x: x[1])[0]

def crossover(parent1, parent2):
    offspring = nx.Graph()
    offspring.add_nodes_from(parent1)
    # Convert edges to a list before sampling
    edges1 = list(parent1.edges())
    edges2 = list(parent2.edges())
    # Sample half of the edges from each parent to form the offspring
    offspring_edges = random.sample(edges1, len(edges1) // 2) + random.sample(edges2, len(edges2) // 2)
    offspring.add_edges_from(offspring_edges)
    return offspring

def mutate(graph, mutation_rate):
    for edge in list(graph.edges()):
        if random.random() < mutation_rate:
            graph.remove_edge(*edge)
    # Ensure the mutation rate does not attempt to add more edges than possible,
    # which can happen in dense graphs
    max_possible_edges = len(graph.nodes()) * (len(graph.nodes()) - 1) / 2
    current_edges = graph.number_of_edges()
    additional_edges = int(min(len(graph.edges()), max_possible_edges - current_edges) * mutation_rate)
    for _ in range(additional_edges):
        # Convert graph.nodes() to a list before sampling
        n1, n2 = random.sample(list(graph.nodes()), 2)
        # Ensure the edge does not already exist and n1 is not n2 to avoid self-loops
        if not graph.has_edge(n1, n2) and n1 != n2:
            graph.add_edge(n1, n2)
    return graph


def save_graph(graph, generation, fitness, index):
    filename = f"{save_path}/gen_{generation}_idx_{index}_fit_{fitness:.2f}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(graph, f)

def genetic_algorithm():
    population = initialize_population(population_size, number_of_nodes)
    best_fitness = float('-inf')
    best_graph = None

    for generation in range(generations):
        fitnesses = [calculate_fitness(graph) for graph in population]
        new_population = []
        for i in range(population_size // 2):
            parent1 = select_parent(population, fitnesses)
            parent2 = select_parent(population, fitnesses)
            offspring1, offspring2 = parent1, parent2
            if random.random() < crossover_rate:
                offspring1 = crossover(parent1, parent2)
                offspring2 = crossover(parent2, parent1)
            offspring1 = mutate(offspring1, mutation_rate)
            offspring2 = mutate(offspring2, mutation_rate)
            new_population.extend([offspring1, offspring2])

        population = new_population
        generation_best_fitness = max(fitnesses)
        if generation_best_fitness > best_fitness:
            best_fitness = generation_best_fitness
            best_graph_index = fitnesses.index(generation_best_fitness)
            best_graph = population[best_graph_index]
            save_graph(best_graph, generation + 1, best_fitness, best_graph_index)

        print(f"Generation {generation + 1}, Best fitness: {best_fitness}")

genetic_algorithm()
