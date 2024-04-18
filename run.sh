#!/bin/bash

CORADIR="./cora"

# Check if the "cora" directory already exists
if [ ! -d "$CORADIR" ]; then
    echo "Cora directory does not exist. Downloading the dataset..."

    # Commands to download the dataset
    wget https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz

    echo "Extracting the dataset..."
    tar -xzvf cora.tgz

    echo "Cleaning up downloaded files..."
    rm cora.tgz

else
    echo "Cora directory already exists. No need to download."
fi


echo "Starting training..."
python train.py --config_path configs/test_config.toml

echo "Starting prediction..."
python predict.py --config_path configs/pred_config.toml