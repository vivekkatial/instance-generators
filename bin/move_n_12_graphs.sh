#!/bin/bash

echo "Starting script..."

# Step 1: Check if the directory final_population_n_12 exists; if not, create it
if [ ! -d "final_population_n_12" ]; then
    echo "Directory final_population_n_12 not found. Creating it..."
    mkdir final_population_n_12
else
    echo "Directory final_population_n_12 already exists."
fi

# Step 2: Copy .pkl files from directories not ending in "n_{integer}" and append the directory name to the copied files
echo "Processing directories and copying .pkl files..."
for dir in target-point-graphs/target_point_*; do
    # Skip directories ending with "n_{integer}"
    if [[ ! $dir =~ n_([0-9]+)$ ]]; then
        # Extract the directory name to use in the file rename
        dir_name=$(basename "$dir")
        echo "Processing directory: $dir_name"

        # Initialize a counter for copied files
        copied_files_count=0

        # Copy and rename .pkl files
        for file in "$dir"/*.pkl; do
            if [ -f "$file" ]; then
                file_name=$(basename "$file")
                # Append directory name to the file name, excluding extension, and copy to the final_population_n_12 directory
                cp "$file" "final_population_n_12/${dir_name}_${file_name}"
                let copied_files_count++
            fi
        done

        if [ $copied_files_count -gt 0 ]; then
            echo "Copied $copied_files_count .pkl files from $dir_name to final_population_n_12."
        else
            echo "No .pkl files found in $dir_name to copy."
        fi
    fi
done

echo "Script execution completed."
