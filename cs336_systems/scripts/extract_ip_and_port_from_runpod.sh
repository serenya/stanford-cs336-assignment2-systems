#!/bin/bash

output=$(runpodctl get pod $1 --allfields)

echo "Pod details: $output"

# Define the input string
INPUT_STRING="$output"

# --- 1. Extraction using grep with PCRE ---
# This step extracts the complete IP:Port string (e.g., "195.26.232.152:45669")
# -o: Only print the matching part.
# -P: Enable Perl-compatible regular expressions.
# Regex: [\d.:]+(?=->22) captures IP:Port only if followed by ->22.
FULL_ADDRESS=$(echo "$INPUT_STRING" | grep -oP '[\d.:]+(?=->22)')

# --- 2. Separation using shell parameter expansion ---
# Extract the IP address (everything before the last ':')
# ${FULL_ADDRESS%:*} removes the shortest match of ':*' from the end of the string.
export IP="${FULL_ADDRESS%:*}"

# Extract the Port number (everything after the first ':')
# ${FULL_ADDRESS#*:} removes the shortest match of '*:' from the start of the string.
export PORT="${FULL_ADDRESS#*:}"
