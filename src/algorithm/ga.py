import networkx as nx
import random
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# print the current working directory
print(os.getcwd())

from src.preprocess import get_z1_z2_projection
from src.generate_planar_graphs import generate_planar_graph_with_min_edges
from src.graph_instance import GraphInstance


def initialise_population(
    population_size, number_of_nodes, min_edges, graph_types, **kwargs
):
    """
    Initialise a population of graphs using a diverse set of graph generation methods.
    """

    # Check kwargs if mutation
    if 'immigration' in kwargs:
        immigration = kwargs['immigration']
    else:
        immigration = False

    population = []

    for graph_type in graph_types:
        for i in range(population_size):
            while True:
                if graph_type == 'planar':
                    edges_to_generate = random.randint(min_edges, 3 * number_of_nodes - 6)
                    G = generate_planar_graph_with_min_edges(number_of_nodes, edges_to_generate)

                elif graph_type == 'erdos_renyi':
                    p = 0.5  # Probability of edge creation
                    G = nx.erdos_renyi_graph(number_of_nodes, p)

                elif graph_type == 'powerlaw_cluster':
                    m = 3  # Number of random edges to add for each new node
                    p = 0.5  # Probability of adding a triangle after adding a random edge
                    G = nx.powerlaw_cluster_graph(number_of_nodes, m, p)

                elif graph_type == 'powerlaw_tree':
                    G = nx.random_powerlaw_tree(number_of_nodes, tries=10000)
                    if G.number_of_edges() < min_edges:
                        continue

                elif graph_type == 'nearly_complete_bipartite':
                    n_part_1 = random.randint(1, number_of_nodes - 1)
                    n_part_2 = number_of_nodes - n_part_1
                    G_temp = nx.complete_bipartite_graph(n_part_1, n_part_2)
                    G = GraphInstance(G_temp, "Nearly Complete BiPartite")
                    G.nearly_complete()
                    G = G.G

                elif graph_type == 'three_regular_graph':
                    G = nx.random_regular_graph(3, number_of_nodes)

                elif graph_type == "four_regular_graph":
                    G = nx.random_regular_graph(4, number_of_nodes)

                elif graph_type == 'geometric':
                    random_radius = random.uniform(0.24, 1)
                    G = nx.random_geometric_graph(number_of_nodes, radius=random_radius)

                if nx.is_connected(G) and G.number_of_edges() >= min_edges:
                    break

            # Check if the generated graph meets the minimum edges criteria before adding
            if G.number_of_edges() >= min_edges:
                population.append((G, graph_type))

            # Apply a weight to each graph type based on the number of graphs of that type
            weight_types = ['cauchy', 'exponential', 'log-normal', 'normal', 'uniform', 'uniform_plus']

            # Randomly select one of those weight_types
            weight_type = random.choice(weight_types)

            # Assign normalized random weights to all edges in the graph based on the specified weight type, unless weight_type is None.
            if weight_type is not None:
                if weight_type == 'uniform':
                    weights = [random.uniform(0, 1) for _ in G.edges()]
                elif weight_type == 'uniform_plus':
                    weights = [random.uniform(-1, 1) for _ in G.edges()]
                elif weight_type == 'normal':
                    weights = [random.normalvariate(0, 1) for _ in G.edges()]
                elif weight_type == 'exponential':
                    weights = [random.expovariate(0.2) for _ in G.edges()]
                elif weight_type == 'log-normal':
                    weights = [random.lognormvariate(0, 1) for _ in G.edges()]
                elif weight_type == 'cauchy':
                    weights = [np.random.standard_cauchy() for _ in G.edges()]
                else:
                    weights = [1 for _ in G.edges()]
                    print("No valid weight type specified; assigning all weights as 1.")

                # Normalize weights to the range [0, 1]
                min_weight = min(weights)
                max_weight = max(weights)
                if max_weight != min_weight:
                    normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in weights]
                else:
                    normalized_weights = [0.5 for _ in weights]  # Default to middle if all weights are the same

                # Assign normalized weights to each edge
                for (u, v), weight in zip(G.edges(), normalized_weights):
                    G[u][v]['weight'] = weight

            # Check if the graph is connected
            if not nx.is_connected(G):
                print(f"Graph is not connected: {graph_type}")
                print(f"Number of nodes: {G.number_of_nodes()}")
                print(f"Number of edges: {G.number_of_edges()}")
                print(f"Minimum edges: {min_edges}")
                print(f"Edges: {G.edges()}")
                raise ValueError("Graph is not connected")

    # Ensure the population does not exceed the desired population size after adding all types
    population = random.sample(population, min(population_size, len(population)))

    # Print distribution of graph types in the population only if not mutation
    if not immigration:
        print(f"Population distribution: {len(population)} graphs")
        for graph_type in graph_types:
            count = sum([1 for g in population if g[1] == graph_type])
            print(f"{graph_type}: {count} graphs")
    # Only return the graphs, not the graph types
    population = [graph for graph, _ in population]

    return population


def fitness_function(graph, target_point, experiment):
    """Fitness function for the genetic algorithm."""
    # Get the Z1, Z2 projection of the graph
    z1_z2_projection = get_z1_z2_projection(graph, experiment=experiment)
    # If (nan, nan) is returned, raise an error
    if np.isnan(z1_z2_projection).all():
        raise ValueError("Projection returned NaN values")
    # Calculate the distance from the target point
    distance = np.linalg.norm(z1_z2_projection - target_point)
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
    rest_of_population = selected_population(
        population, fitness_scores, len(population) - elite_size
    )

    # Combine elites with the rest of the newly generated population
    new_population = elites + rest_of_population

    return new_population


def elitist_selection(population, fitness_scores, elite_size):
    # Ensure elite_size does not exceed the population size
    elite_size = min(elite_size, len(population))

    # Indices of sorted fitness scores (ascending order since lower is better)
    sorted_indices = np.argsort(fitness_scores)

    # Directly select the top-performing individuals based on elite_size
    elites = [population[i] for i in sorted_indices[:elite_size]]

    # The rest of the selection pool
    non_elites = [population[i] for i in sorted_indices[elite_size:]]
    non_elite_scores = [fitness_scores[i] for i in sorted_indices[elite_size:]]

    # Invert fitness scores for selection probability calculation (assuming lower is better)
    inverted_scores = [1 / score for score in non_elite_scores]
    total_inverted = sum(inverted_scores)
    selection_probabilities = [score / total_inverted for score in inverted_scores]

    # Select individuals to fill the remaining population spots
    remaining_selections = len(population) - elite_size
    if remaining_selections > 0:
        selected_indices = np.random.choice(
            range(len(non_elites)),
            size=remaining_selections,
            replace=True,
            p=selection_probabilities,
        )
        selected_non_elites = [non_elites[i] for i in selected_indices]
    else:
        selected_non_elites = []

    # Combine elites and selected non-elites for the new population
    new_population = elites + selected_non_elites

    return new_population


def structured_graph_crossover(
    parent1, parent2, number_of_nodes, min_edges, mutation_rate
):
    """
    Performs a crossover between two parent graphs by prioritizing edges that are common
    to both parents and then selectively choosing edges from each parent to ensure the
    offspring is connected and adheres to structural similarities with the parents.

    Parameters:
    - parent1, parent2: NetworkX Graph objects (the parents)
    - number_of_nodes: The number of nodes in each graph (assumed equal for both parents)
    - min_edges: Minimum number of edges the offspring graph must have
    - mutation_rate: Float, the probability of introducing a mutation in the offspring

    Returns:
    - offspring: A new NetworkX Graph object resulting from the crossover
    """
    # Initialize an empty graph for the offspring
    offspring = nx.Graph()
    offspring.add_nodes_from(range(number_of_nodes))

    # Identify common edges and unique edges in both parents
    common_edges = set(parent1.edges()) & set(parent2.edges())
    unique_edges = (set(parent1.edges()) - common_edges) | (
        set(parent2.edges()) - common_edges
    )

    # First, add all common edges to the offspring to preserve structural features
    offspring.add_edges_from(common_edges)

    # If not fully connected, add edges from the unique set to enhance connectivity
    unique_edges_list = list(unique_edges)
    # Shuffle to introduce some variability in the offspring
    random.shuffle(unique_edges_list)
    for edge in unique_edges_list:
        if nx.is_connected(offspring):
            break  # Stop once we achieve a connected graph
        offspring.add_edge(*edge)

    # Introduce mutation with a given probability
    if random.random() < mutation_rate:
        potential_edges = [
            (i, j)
            for i in range(number_of_nodes)
            for j in range(i + 1, number_of_nodes)
            if not offspring.has_edge(i, j)
        ]

        add_or_remove = random.choice([True, False])

        if add_or_remove and potential_edges:
            edge = random.choice(potential_edges)
            offspring.add_edge(*edge)
        elif not add_or_remove and offspring.number_of_edges() > min_edges:
            edge = random.choice(list(offspring.edges()))
            offspring.remove_edge(*edge)

    # Ensure the offspring meets the minimum edge requirement
    while offspring.number_of_edges() < min_edges:
        potential_edges = [
            (i, j)
            for i in range(number_of_nodes)
            for j in range(i + 1, number_of_nodes)
            if not offspring.has_edge(i, j)
        ]
        if (
            not potential_edges
        ):  # If no more edges can be added, break to avoid infinite loop
            break
        edge = random.choice(potential_edges)
        offspring.add_edge(*edge)

    # Finally, ensure the graph is connected; if not, add edges until it is
    while not nx.is_connected(offspring):
        potential_edges = [
            (i, j)
            for i in range(number_of_nodes)
            for j in range(i + 1, number_of_nodes)
            if not offspring.has_edge(i, j)
        ]
        edge = random.choice(potential_edges)
        offspring.add_edge(*edge)

    return offspring


def immigrate_graph_population(
    population, immigration_rate, number_of_nodes, min_edges, graph_types
):
    """
    Introduces immigrants into the graph population with a given probability.
    Immigrants are generated using the 'initialise_population' function, simulating
    the introduction of new individuals from an external source.

    Parameters:
    - population: List of NetworkX graph objects representing the current population.
    - immigration_rate: Float, the probability of introducing an immigrant graph into the population.
    - number_of_nodes: Int, the number of nodes for new immigrant graphs.
    - min_edges: Int, the minimum number of edges for new immigrant graphs.
    - graph_types: List of strings, the types of graphs that can be generated as immigrants.

    Returns:
    - A new population with introduced immigrants.
    """
    new_population = []
    for graph in population:
        if random.random() < immigration_rate:
            # Immigration: Introduce a new graph from the external source (initialise_population)
            new_graphs = initialise_population(
                1, number_of_nodes, min_edges, graph_types=graph_types, immigration=True
            )
            assert len(new_graphs) == 1, "Expected a single graph for immigration"
            new_population.append(new_graphs[0])
        else:
            # No immigration, keep the original graph
            new_population.append(graph)
    return new_population


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Run the genetic algorithm.")
    parser.add_argument(
        "--population_size",
        type=int,
        default=20,
        help="The size of the population for the genetic algorithm.",
    )
    parser.add_argument(
        "--number_of_nodes",
        type=int,
        default=12,
        help="The number of nodes for the graphs in the population.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=10000,
        help="The number of generations for the genetic algorithm.",
    )
    parser.add_argument(
        "--mutation_rate",
        type=float,
        default=0.05,
        help="The mutation rate for the genetic algorithm.",
    )
    parser.add_argument(
        "--immigration_rate",
        type=float,
        default=0.05,
        help="The immigration rate for the genetic algorithm.",
    )
    parser.add_argument(
        "--elite_size",
        type=int,
        default=6,
        help="The number of elites to keep in the population.",
    )
    parser.add_argument(
        "--target_point",
        type=float,
        nargs=2,
        default=[2.5, 2.5],
        help="The target point for the fitness function.",
    )
    parser.add_argument(
        "--min_edges",
        type=int,
        default=11,
        help="The minimum number of edges for graphs in the population.",
    )

    parser.add_argument(
        "--experiment",
        type=str,
        nargs=1,
        default="INFORMS-Revision-12-node-network",
        help="The QAOA expeirment.",
    )

    args = parser.parse_args()

    # Unpack the arguments
    population_size = args.population_size
    number_of_nodes = args.number_of_nodes
    generations = args.generations
    mutation_rate = args.mutation_rate
    immigration_rate = args.immigration_rate
    elite_size = args.elite_size
    target_point = args.target_point
    min_edges = args.min_edges
    experiment = args.experiment

    graph_types = [
        'planar',
        'erdos_renyi',
        'powerlaw_cluster',
        'powerlaw_tree',
        'nearly_complete_bipartite',
        'geometric',
        'three_regular_graph',
        'four_regular_graph',
    ]

    # Make save path based on target-point
    save_path = os.path.join(
        experiment,
        "target-point-graphs", 
        f"target_point_{target_point[0]}_{target_point[1]}_n_{number_of_nodes}"
    )
    

    print('=========================================================================')
    print('-> Kicking off Genetic Algorithm.')
    print(f'-> Results will be saved to: {save_path}')
    print('=========================================================================')

    # Make sure the save directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Initialisation and evaluation of the initial population
    initial_population = initialise_population(
        population_size, number_of_nodes, min_edges, graph_types
    )

    fitness_scores = [
        fitness_function(graph, target_point, experiment=experiment) for graph in initial_population
    ]
    avg_fitness = np.mean(fitness_scores)
    best_fitness = np.min(fitness_scores)
    print(f"Average fitness of initial population: {avg_fitness}")
    print(f"Best fitness of initial population: {best_fitness}")

    best_fitnesses = [best_fitness]
    avg_fitnesses = [avg_fitness]
    generation = [0]

    # Save the best graph in the initial population
    best_graph_index = np.argmin(fitness_scores)
    best_graph = initial_population[best_graph_index]
    path = os.path.join(save_path, "best_graph_gen_0.graphml")
    nx.write_graphml(best_graph, path)

    # Integrate elitism: Select top-performing individuals (elites) and rest of the population for breeding
    selected_population = elitist_selection(
        initial_population, fitness_scores, elite_size
    )

    # Set up a stopping criterion
    stop_criterion_met = False

    # Main genetic algorithm loop
    for gen in range(generations):
        print(f"Generation {gen+1}/{generations}")

        # Crossover and mutation operations
        offspring_population = [
            structured_graph_crossover(
                random.choice(selected_population),
                random.choice(selected_population),
                number_of_nodes,
                min_edges,
                mutation_rate,
            )
            for _ in range(population_size - elite_size)
        ]

        immigrated_population = immigrate_graph_population(
            offspring_population, mutation_rate, number_of_nodes, min_edges, graph_types
        )

        # Combine and select the new population
        combined_population = immigrated_population + selected_population[:elite_size]
        fitness_scores = [
            fitness_function(graph, target_point, experiment=experiment) for graph in combined_population
        ]
        selected_population = elitist_selection(
            combined_population,
            fitness_scores,
            elite_size,
        )

        # Re-evaluate fitness scores for the selected population to ensure alignment
        updated_fitness_scores = [
            fitness_function(graph, target_point, experiment=experiment) for graph in selected_population
        ]

        # Update metrics based on the updated fitness scores
        avg_fitness = np.mean(updated_fitness_scores)
        best_fitness = np.min(updated_fitness_scores)
        best_fitnesses.append(best_fitness)
        avg_fitnesses.append(avg_fitness)
        generation.append(gen + 1)

        print(f"Average fitness of new population: {avg_fitness}")
        print(f"Best fitness of new population: {best_fitness}")

        # Find the best graph in the selected population
        best_graph_index = np.argmin(updated_fitness_scores)
        best_graph = selected_population[best_graph_index]
        
        if gen >= 300:
            best_fitnesses_last_300 = best_fitnesses[-300:]
            best_fitnesses_last_300_min = np.min(best_fitnesses_last_300)
            best_fitnesses_last_300_max = np.max(best_fitnesses_last_300)
            # If the difference is less than 0.01% of the best fitness, stop and store results
            if best_fitnesses_last_300_max - best_fitnesses_last_300_min < 0.0001 * best_fitnesses_last_300_max:
                print("Stopping criterion met: Best fitness has not improved by more than 0.1% in the last 300 generations.")
                stop_criterion_met = True
                path = os.path.join(save_path, f"best_graph_gen_{gen+1}.graphml")
                nx.write_graphml(best_graph, path)
                break  # Stops the loop
    
    # If stopping criterion not met, save the best graph in the final population
    if not stop_criterion_met:
        path = os.path.join(save_path, f"best_graph_gen_{generations}.graphml")
        nx.write_graphml(best_graph, path)

    # Save performance metrics
    performance_metrics = {
        "generation": generation,
        "best_fitness": best_fitnesses,
        "avg_fitness": avg_fitnesses,
    }
    performance_metrics_df = pd.DataFrame(performance_metrics)
    print(performance_metrics_df)
    # Write to CSV
    performance_metrics_df.to_csv(
        os.path.join(save_path, "performance_metrics.csv"), index=False
    )

    # Set the Seaborn theme for aesthetics
    sns.set_theme(
        context='paper',
        style='whitegrid',
        palette='muted',
        font='serif',
        font_scale=1.2,
    )

    # Create the plot
    plt.figure(figsize=(12, 6), dpi=300)

    # Plotting both 'best_fitness' and 'avg_fitness' against 'generation'
    sns.lineplot(
        data=performance_metrics_df,
        x='generation',
        y='best_fitness',
        label='Best Fitness',
        linewidth=2,
    )
    sns.lineplot(
        data=performance_metrics_df,
        x='generation',
        y='avg_fitness',
        label='Average Fitness',
        linewidth=2,
    )

    # Enhancing the plot
    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Fitness", fontsize=14)
    plt.title("Fitness over Generations", fontsize=16, weight='bold')
    plt.legend(title='Metric', title_fontsize='13', fontsize='12')
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(save_path, "fitness_over_generations.png"))

    # Save all the graphs in the final population
    for i, graph in enumerate(selected_population):
        path = os.path.join(save_path, f"final_population_graph_{i}.graphml")
        nx.write_graphml(graph, path)
