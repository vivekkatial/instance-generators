#!/bin/bash

# Define the input CSV file
INPUT_FILE="qaoa-param-evolved/final_evolved_instances_n_12_with_source.csv"

# Skip the header row
tail -n +2 "$INPUT_FILE" | awk -F, '{
    # Prefix the source file location and construct the scp command
    # The source file path is now "qaoa-param-evolved/best_graphs_12/" prepended to the value in the 3rd column (Source)
    # The destination path is constructed similarly, but based on the value in the 4th column (Source_new)
    cmd = "scp \"qaoa-param-evolved/best_graphs_12/" $3 "\" spartan.hpc.unimelb.edu.au:/data/cephfs/punim1074/HAQC/data/\"" $4 "\""
    # Execute the scp command
    system(cmd)
}'
