#!/bin/bash

# Create the pod and capture the line containing the pod id
echo -e '\e[1;33mStep 0 - spin up RunPod instance\e[0m'

PUBLIC_KEY=$(<"$HOME/.ssh/runpod_ed25519.pub")

output=$(runpodctl create pod \
  --name "my-training-pod" \
  --templateId "runpod-torch-v280" \
  --gpuType "NVIDIA L40S" \
  --gpuCount 1 \
  --secureCloud \
  --imageName "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404" \
  --containerDiskSize 30 \
  --volumeSize 10 \
  --env PUBLIC_KEY="$PUBLIC_KEY")

echo "create output: $output"

# Extract the pod ID using regex (assuming output: pod "<ID>" created ...)
export POD_ID=$(echo "$output" | grep -oP 'pod "\K[^"]+')