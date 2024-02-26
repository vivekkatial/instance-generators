#!/bin/bash

# Create a directory called 'best_graphs' to store the copied files
mkdir -p best_graphs

# Find all .pkl files in subdirectories and copy them to the 'best_graphs' directory
# with their target point coordinates included in the filename
find . -name "*.pkl" -exec bash -c '{
    # Get the full file path
    file_path="$1"
    
    # Extract the base filename
    file_name=$(basename "$file_path")
    
    # Extract the directory name to use in the new filename
    dir_name=$(dirname "$file_path")
    dir_name=$(basename "$dir_name")
    
    # Copy the file to the 'best_graphs' directory with the new name
    cp "$file_path" "best_graphs/${dir_name}_$file_name"
}' bash {} \;

echo "All .pkl files have been copied to the best_graphs directory."
