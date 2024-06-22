#!/bin/bash

# Define the source and destination directories
src_dir="."
dest_dir="./best_graphs_n_16"

# Check if the destination directory exists, if not, create it
if [ ! -d "$dest_dir" ]; then
    mkdir -p "$dest_dir"
fi

# Find directories with _n_16 in their names under any target_point_* directory and iterate over them
find "$src_dir" -type d -regex ".*/target_point_.*_n_16" | while read dir; do
    # Extract the target_point part from the directory path for use in the new filename
    target_point=$(basename "$dir")
    # Inside each directory, look for pkl files containing 'final_population' in their names
    find "$dir" -type f -name "*final_population*.graphml" | while read file; do
        # Construct new filename with target_point preserved
        new_filename="${dest_dir}/${target_point}_$(basename "$file")"
        # Copy each found file to the destination directory with the new filename
        cp "$file" "$new_filename"
        echo "Copied $file to $new_filename"
    done
done
