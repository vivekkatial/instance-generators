import networkx as nx
import random
import numpy as np
import os
import pickle
import pandas as pd 
import matplotlib.pyplot as plt
from preprocess_manual import get_z1_z2_projection
from generate_planar_graphs import generate_planar_graph_with_min_edges
from graph_instance import GraphInstance

# Parameters
population_size = 20
mutation_rate = 0.02
crossover_rate = 0.5
number_of_nodes = 12  # This might become a range if you want to evolve this too
target_point = [2.5, 2.5]
generations = 100
save_path = "best_graphs"  # Directory to save best parameter sets

def initialisation_population(population_size, number_of_nodes, min_edges):
    population = []
    for i in range(population_size):
        # Generate planar graph with random number between min_edges and 3 * number_of_nodes - 6
        edges_to_generate = random.randint(min_edges, 3 * number_of_nodes - 6)
        G = generate_planar_graph_with_min_edges(number_of_nodes, edges_to_generate)
        population.append(G)
    for i in range(population_size):
        G = nx.erdos_renyi_graph(number_of_nodes, 0.5)
        population.append(G)
    for i in range(population_size):
        G = nx.powerlaw_cluster_graph(number_of_nodes, 3, 0.5)
        population.append(G)
    for i in range(population_size):
        G = nx.random_regular_graph(3, number_of_nodes)
        population.append(G)
    for i in range(population_size):
        G = nx.random_tree(number_of_nodes)
        population.append(G)
    for i in range(population_size):
         # Create a nearly compelte bi partite graph
        # Randomly generate the size of one partiton
        n_part_1 = random.randint(1, number_of_nodes - 1)
        n_part_2 = number_of_nodes - n_part_1
        # Generate nearly complete bipartite graph
        G_nc_bipart = GraphInstance(
            nx.complete_bipartite_graph(n_part_1, n_part_2), "Nearly Complete BiPartite"
        )
        G_nc_bipart.nearly_complete()
        population.append(G_nc_bipart.G)
    
    # Remove graphs with less than min_edges
    population = [G for G in population if G.number_of_edges() >= min_edges]

    # Randomly sample population_size graphs from the population
    population = random.sample(population, population_size)
    return population

def fitness_function(graph, target_point):
    """Fitness function for the genetic algorithm."""
    # Get the Z1, Z2 projection of the graph
    z1_z2_projection = get_z1_z2_projection(graph)
    # Calculate the distance from the target point
    distance = np.linalg.norm(z1_z2_projection-target_point)
    return distance

def integrate_elitism(population, fitness_scores, elite_size):
    """
    Integrate elitism into the genetic algorithm by preserving a subset of the best individuals
    from the population unchanged for the next generation.
    
    Parameters:
    - population: List of individuals in the current generation.
    - fitness_scores: Corresponding fitness scores for the population.
    - elite_size: Number of top individuals to carry over unchanged.
    
    Returns:
    - new_population: Population for the next generation with elites included.
    """
    # Sort the population by fitness scores, assuming lower scores are better
    sorted_indices = np.argsort(fitness_scores)
    
    # Select the top `elite_size` individuals
    elites = [population[i] for i in sorted_indices[:elite_size]]
    
    # Generate the rest of the population through selection, crossover, and mutation
    # This is a placeholder for the actual genetic operations you might perform
    rest_of_population = selected_population(population, fitness_scores, len(population) - elite_size)
    
    # Combine elites with the rest of the newly generated population
    new_population = elites + rest_of_population
    
    return new_population


def elitist_selection(population, fitness_scores, elite_size):
    """
    Select parents for the next generation, explicitly preserving a specified number
    of top-performing individuals (elites) and using a probabilistic selection for the rest.
    """
    # Ensure elite_size does not exceed the population size
    elite_size = min(elite_size, len(population))
    
    # Indices of sorted fitness scores (ascending order since lower is better)
    sorted_indices = np.argsort(fitness_scores)
    
    # Directly select the top-performing individuals based on elite_size
    elites = [population[i] for i in sorted_indices[:elite_size]]
    
    # The rest of the selection pool
    non_elites = [population[i] for i in sorted_indices[elite_size:]]
    non_elite_scores = [fitness_scores[i] for i in sorted_indices[elite_size:]]
    
    # Calculate selection probabilities for the non-elites
    # Normalize fitness scores for probability calculation
    total_fitness = sum(non_elite_scores)
    selection_probabilities = [score / total_fitness for score in non_elite_scores]
    
    # Select individuals to fill the remaining population spots
    remaining_selections = len(population) - elite_size
    if remaining_selections > 0:
        selected_indices = np.random.choice(range(len(non_elites)), size=remaining_selections, replace=True, p=selection_probabilities)
        selected_non_elites = [non_elites[i] for i in selected_indices]
    else:
        selected_non_elites = []

    # Combine elites and selected non-elites for the new population
    new_population = elites + selected_non_elites
    
    return new_population



def graph_crossover(parent1, parent2, number_of_nodes, min_edges):
    """
    Performs a simple crossover between two parent graphs by randomly selecting
    edges from each parent to produce an offspring. This implementation ensures
    the offspring is connected.
    
    Parameters:
    - parent1, parent2: NetworkX Graph objects (the parents)
    - number_of_nodes: The number of nodes in each graph (assumed equal for both parents)
    
    Returns:
    - offspring: A new NetworkX Graph object resulting from the crossover
    """
    # Initialize an empty graph for the offspring
    offspring = nx.Graph()
    offspring.add_nodes_from(range(number_of_nodes))
    
    # Get edges from both parents
    edges1 = list(parent1.edges())
    edges2 = list(parent2.edges())
    
    # Randomly shuffle the edges to ensure a mix in the offspring
    random.shuffle(edges1)
    random.shuffle(edges2)
    
    # Combine the edges, ensuring no duplicates
    combined_edges = list(set(edges1 + edges2))
    
    # Add edges to the offspring from the combined list until it's fully connected
    while not nx.is_connected(offspring):
        if not combined_edges:  # If we run out of edges before connecting the graph, break to avoid infinite loop
            break
        edge = combined_edges.pop(0)
        offspring.add_edge(*edge)
        # Check for connectivity and remove any cycles if necessary
        if len(nx.cycle_basis(offspring)) > 0:
            offspring.remove_edge(*edge)
    
    # Check if the offspring meet minimum edge requirements
    if offspring.number_of_edges() < min_edges:
        # If not, add random edges until the minimum is reached
        while offspring.number_of_edges() < min_edges:
            edge = random.sample(range(number_of_nodes), 2)
            offspring.add_edge(*edge)

    return offspring

def mutate_graph_population(population, mutation_rate, number_of_nodes, min_edges):
    """
    Mutates each graph in the population with a given probability. Mutation is a 
    randomly generated graph from a diverse set of graph types.

    Parameters:
    - population: List of NetworkX graph objects representing the current population.
    - mutation_rate: Float, the probability of mutating a given graph in the population.
    - number_of_nodes: Int, the number of nodes for new graphs if a mutation involves generating a new graph.
    - min_edges: Int, the minimum number of edges for new graphs if a mutation involves generating a new graph.

    Returns:
    - A new population with mutated graphs.
    """
    mutated_population = []
    for graph in population:
        if random.random() < mutation_rate:
            # Mutation: Replace with a new graph from the diverse population methods
            new_graphs = initialisation_population(1, number_of_nodes, min_edges)
            assert len(new_graphs) == 1, "Expected a single graph"
            mutated_population.append(new_graphs[0])
        else:
            # No mutation, keep the original graph
            mutated_population.append(graph)
    return mutated_population



if __name__ == "__main__":
    # Configuration parameters
    population_size = 100  # Example size
    number_of_nodes = 50  # Example nodes
    generations = 50  # Number of generations
    mutation_rate = 0.1  # Mutation rate
    elite_size = 20  # Number of elites to keep
    target_point = 0.5  # Example target for fitness function
    min_edges = 23  # Minimum number of edges for graphs
    save_path = './save_graphs'  # Path to save best graphs
    
    # Make sure the save directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Initialisation and evaluation of the initial population
    initial_population = initialisation_population(population_size, number_of_nodes, min_edges)
    fitness_scores = [fitness_function(graph, target_point) for graph in initial_population]
    avg_fitness = np.mean(fitness_scores)
    best_fitness = np.min(fitness_scores)
    print(f"Average fitness of initial population: {avg_fitness}")
    print(f"Best fitness of initial population: {best_fitness}")
    
    best_fitnesses = [best_fitness]
    avg_fitnesses = [avg_fitness]
    generation = [0]

    # Integrate elitism: Select top-performing individuals (elites) and rest of the population for breeding
    selected_population = elitist_selection(initial_population, fitness_scores, elite_size)

    # Main genetic algorithm loop
    for gen in range(generations):
        print(f"Generation {gen+1}/{generations}")

        # Crossover and mutation operations
        offspring_population = [graph_crossover(random.choice(selected_population), random.choice(selected_population), number_of_nodes, min_edges) for _ in range(population_size - elite_size)]
        mutated_population = mutate_graph_population(offspring_population, mutation_rate, number_of_nodes, min_edges)

        # Re-evaluate fitness for the new population
        fitness_scores = [fitness_function(graph, target_point) for graph in mutated_population + selected_population[:elite_size]]
        selected_population = elitist_selection(mutated_population + selected_population[:elite_size], fitness_scores, elite_size)

        # Update metrics
        avg_fitness = np.mean(fitness_scores)
        best_fitness = np.min(fitness_scores)
        best_fitnesses.append(best_fitness)
        avg_fitnesses.append(avg_fitness)
        generation.append(gen+1)

        print(f"Average fitness of new population: {avg_fitness}")
        print(f"Best fitness of new population: {best_fitness}")

        # Save the best graph of this generation
        best_graph_index = np.argmin(fitness_scores)
        best_graph = selected_population[best_graph_index]
        path = os.path.join(save_path, f"best_graph_{gen+1}.pkl")
        with open(path, 'wb') as file:
            pickle.dump(best_graph, file)

    # Save performance metrics
    performance_metrics = {"generation": generation, "best_fitness": best_fitnesses, "avg_fitness": avg_fitnesses}
    performance_metrics_df = pd.DataFrame(performance_metrics)
    print(performance_metrics_df)
    # Plot best fitness and average fitness over generations
    plt.plot(generation, best_fitnesses, label="Best Fitness")
    plt.plot(generation, avg_fitnesses, label="Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness over Generations")
    plt.legend()
    plt.show()
