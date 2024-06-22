#!/bin/bash

# Base directory for the operation, assumed to be the root directory you're working from
base_dir="qaoa-param-evolved"

# Target directory containing the subdirectories
target_dir="${base_dir}/target-point-graphs"

# Ensure the base directory for best graphs exists in the root
mkdir -p "${base_dir}/best_graphs_14"

# Iterate over each subdirectory in target-point-graphs
for dir in ${target_dir}/target_point_*; do
    # Extract the target_point identifier from the directory name
    target_point=$(basename "$dir")
    # Check if the directory name contains '_n_'
    if [[ $dir =~ _n_([0-9]+) ]]; then
        # Extract node size from directory name
        node_size=${BASH_REMATCH[1]}
    else
        # Default node size for directories not specifying a node size
        node_size=12
    fi

    # Ensure the target directory for best graphs exists in the root
    mkdir -p "${base_dir}/best_graphs_${node_size}"

    # Initialize variables to track the highest gen value and corresponding file name
    highest_gen=-1
    best_graph_file=""

    # Iterate over files in the current directory
    for file in ${dir}/best_graph_gen_*.graphml; do
        # Extract the filename from the path
        filename=$(basename "$file")
        # Extract gen value from file name using parameter expansion
        file_gen="${filename#best_graph_gen_}"
        file_gen="${file_gen%.graphml}"

        # Only proceed if file_gen is a number
        if [[ $file_gen =~ ^[0-9]+$ ]]; then
            # Check if this gen is higher than the highest recorded gen
            if [ "$file_gen" -gt "$highest_gen" ]; then
                highest_gen=$file_gen
                best_graph_file=$file
            fi
        fi
    done

    # Copy the file with the highest gen to the target directory in the root
    if [ -n "$best_graph_file" ]; then
        # Append the target_point to the destination filename
        destination_file="${base_dir}/best_graphs_${node_size}/${target_point}_best_graph_gen_${highest_gen}.graphml"
        cp "$best_graph_file" "$destination_file"
    fi
done
