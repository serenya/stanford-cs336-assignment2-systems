#!/bin/bash

source ./spin_runpod_instance.sh

echo "Created Pod ID is: $POD_ID"

source ./extract_ip_and_port_from_runpod.sh $POD_ID

echo "Pod IP: $IP"
echo "Pod Port: $PORT"

ssh root@$IP -p $PORT -i ~/.ssh/runpod_ed25519 \
"bash -s -- https://github.com/serenya/stanford-cs336-assignment2-systems.git task/memory_profiling cs336_systems.benchmarking_script" \
 < run_benchmarking_script_on_runpod.sh

#runpodctl remove pod $POD_ID