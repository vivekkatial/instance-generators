#!/bin/bash

# Define the paths
SOURCE_DIR="INFORMS-Revision-12-node-network/target-point-graphs"
TARGET_DIR="INFORMS-Revision-12-node-network/all-evolved-instances"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Find all .graphml files and copy them with a concatenated name
find "$SOURCE_DIR" -type f -name "*.graphml" | while read file; do
    # Get the directory name
    dir_name=$(basename "$(dirname "$file")")
    # Get the file name
    file_name=$(basename "$file")
    # Construct the new file name
    new_file_name="${dir_name}_${file_name}"
    # Copy the file with the new name
    cp "$file" "$TARGET_DIR/$new_file_name"
done

echo "All .graphml files have been copied successfully with concatenated names."
