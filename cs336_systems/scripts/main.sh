#!/bin/bash

source ./spin_runpod_instance.sh

echo "Created Pod ID is: $POD_ID"

source ./extract_ip_and_port_from_runpod.sh $POD_ID

echo "Source IP variable (IP): $IP"
echo "Source Port variable (PORT): $PORT"

ssh root@$IP -p $PORT -i ~/.ssh/runpod_ed25519 \
"bash -s -- https://github.com/serenya/stanford-cs336-assignment2-systems.git task/task/automate-runpod-instance-creation cs336_systems.benchmarking_script" \
 < cs336_systems/run_benchmarking_script_on_runpod.sh

runpodctl remove pod $POD_ID