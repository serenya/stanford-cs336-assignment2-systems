#!/bin/bash

POD_ID=$1
MAX_ATTEMPTS=5
SLEEP_TIME=10 # Seconds to wait between attempts

echo "Starting to poll for port 22 address for Pod ID: ${POD_ID} (Max attempts: ${MAX_ATTEMPTS})"

sleep $(( 2 * $SLEEP_TIME ))

for i in $(seq 1 $MAX_ATTEMPTS); do
    echo "Attempt $i of $MAX_ATTEMPTS..."

    # Execute the command to get the latest pod details
    # The output variable name is set by the user's request.
    output=$(runpodctl get pod "$POD_ID" --allfields)

    echo "Pod details: $output"

    # Note: If the actual runpodctl command fails (non-zero exit code), you might want 
    # to add error handling here, but we proceed with the user's core logic.

    # Define the input string
    INPUT_STRING="$output"

    # --- Extraction using grep with PCRE ---
    # This step extracts the complete IP:Port string (e.g., "195.26.232.152:45669")
    # -o: Only print the matching part.
    # -P: Enable Perl-compatible regular expressions.
    # Regex: [\d.:]+(?=->22) captures IP:Port only if followed by ->22.
    FULL_ADDRESS=$(echo "$INPUT_STRING" | grep -oP '[\d.:]+(?=->22)')

    # Check if the FULL_ADDRESS was found
    if [ -n "$FULL_ADDRESS" ]; then
        echo "SUCCESS: IP and Port found on attempt $i!"
        
        # Optionally, you can now separate the IP and Port as per the previous script:
        export IP="${FULL_ADDRESS%:*}"
        export PORT="${FULL_ADDRESS#*:}"
        
        # Use 'break' to exit the loop immediately
        break
    fi

    # If this is the last attempt and we haven't broken, inform the user
    if [ $i -eq $MAX_ATTEMPTS ]; then
        echo "FAILURE: Max attempts reached ($MAX_ATTEMPTS). IP:Port not found."
        exit 2
    fi
    
    # Wait before the next attempt
    echo "Address not found yet. Sleeping for ${SLEEP_TIME} seconds..."
    sleep $SLEEP_TIME

done
