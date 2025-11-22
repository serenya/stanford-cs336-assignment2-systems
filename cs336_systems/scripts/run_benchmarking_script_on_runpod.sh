#!/bin/bash

# $1 - Git repository URL to clone on remote RunPod server
# $2 - Feature branch to checkout on remote RunPod server
# $3 - Python module to run on remote RunPod server

echo -e '\e[1;33mStep 1 - clone git repository\e[0m'
git clone $1

echo -e '\e[1;33mStep 2 - change directory into cloned repository\e[0m'
url="$1"
folder_name="${url##*/}"
folder_name="${folder_name%.git}"
cd $folder_name/

echo -e '\e[1;33mStep 3 - checkout feature branch to run the benchmark on remote RunPod server\e[0m'
git checkout $2 

echo -e '\e[1;33mStep 4 - create virtual environment and skip download of already installed packages on RunPod server\e[0m'
uv venv --system-site-packages

echo -e '\e[1;33mStep 5 - run benchmark script on remote RunPod server\e[0m'
uv run --no-default-groups python -m $3