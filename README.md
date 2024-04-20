# Node Classification with Graph Convolutional Network (GCN): Cora Dataset

This project implements a Graph Convolutional Network (GCN) for node classification on the Cora dataset. The GCN model leverages node features (embeddings) and graph structure (adjacency matrix) to classify nodes into different categories based on their content and citation relationships.

## Project Structure

### 1. Data Preparation
Data for the [Cora dataset](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz) will be automatically downloaded if not present in a folder named `cora` in the current working directory. The dataset includes papers (nodes), embeddings (features), categories of papers (labels), and citation relationships (edges).

- **Adjacency Matrix Generation:** Create an adjacency matrix using the citation relationships (edges) among the papers.
- **Normalization:** Apply normalization to both the features (embeddings) and the adjacency matrix to standardize data scales.

### 2. Model Architecture
The GCN model takes the embeddings and adjacency matrix as inputs. It is defined with two graph convolutional layers:
- The first layer transforms the input features into a hidden layer with 20 neurons, followed by a dropout layer for regularization.
- The second graph convolutional layer outputs the class scores.

### 3. Training Process
The Cora dataset is split into training and testing sets using 10-Fold stratified sampling to ensure that the label distribution is consistent between the train and test sets, which is crucial given the label imbalance in the dataset. Each split uses 80% of the data for training and 20% for testing.

During training:
- The model updates using only the training index in each batch.
- Training history (loss and accuracy on train and test sets) is saved in the `train_history` folder.
- Plots of training progress are saved in the `train_history_plots` folder.

### 4. Running the Project (Training and Predicting)

#### 4.1 Configuration Setup
All model and training parameters are defined in the `configs/config.toml` file. This configuration file allows for easy adjustment of parameters.

You can run the project code within two different environments: [**Conda**](#42-run-with-conda) or [**Docker**](#43-run-with-docker).

#### 4.2 Run with Conda 

##### 4.2.1 Conda enviroment set up
```bash
conda create -n node_cora python=3.11
conda activate node_cora
pip install --no-cache-dir -r requirements.txt
```
You can execute the project code within the Conda environment using one of two methods: [**Shell scripts**](#422-shell-scripts) or [**Python commands**](#423-python-commands).

##### 4.2.2 Shell scripts 
```bash
chmod +x run.sh
./run.sh
```
##### 4.2.3 Python commands

If the dataset is not already downloaded in the `cora` folder, you can download and extract it using the following commands:

```bash
# Commands to download the dataset
wget https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz

echo "Extracting the dataset..."
tar -xzvf cora.tgz

echo "Cleaning up downloaded files..."
rm cora.tgz
```

```bash
# Training configurations are stored in configs/config.toml, which includes e.g. model hyperparameters. 
python train.py --config_path configs/config.toml

# By default, the dataset is split using a stratified method. However, you can opt for k-fold cross-validation by specifying the `cv_method` parameter. 
# Example command to train with k-fold cross-validation:**
python train.py --config_path configs/config.toml --cv_method "kfold"

# Configuration for predictions, such as the location of the trained model and output file name, is in configs/pred_config.toml.
python predict.py --config_path configs/pred_config.toml
```

#### 4.3 Run with Docker

```bash
docker-compose up --build
```

### 5. Results

After running the project, you can expect the following outputs, which help in evaluating the performance:

#### 5.1 Prediction Outputs
- **Paper Categories Predictions:** The predictions for the categories of papers are saved in a `prediction.tsv` file. This file contains the predicted categories for each paper in the dataset. If you only need the prediction file and you want to skip the training porcess, you can use the trained model located in `models/model_split_1` to make predictions. Here are the instructions:

You can make the predictions using trained model within two different environments: [**Conda**](#5111-conda-environment-set-up) or [**Docker**](#512-run-with-docker).

##### 5.1.1 Run with Conda (shell script)
##### 5.1.1.1 Conda enviroment set up
```bash
conda create -n node_cora python=3.11
conda activate node_cora
pip install --no-cache-dir -r requirements.txt
```
You can make the predictions within the Conda environment using one of two methods: [**Shell scripts**](#5112-shell-script) or [**Python commands**](#5113-python-commands).
##### 5.1.1.2 Shell script
```bash
chmod +x run.predict.sh
./run.predict.sh
```
##### 5.1.1.3 Python commands
```bash
# Commands to download the dataset
wget https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz

echo "Extracting the dataset..."
tar -xzvf cora.tgz

echo "Cleaning up downloaded files..."
rm cora.tgz
```

```bash
python predict.py --config_path configs/pred_config.toml
```

##### 5.1.2 Run with Docker

```bash
docker-compose -f docker-compose.predict.yaml up --build
```
#### 5.2 Trained Models

- **Model Files:** The trained models for each data split are saved in the `models` folder.

#### 5.3 Training History

- **Loss and Accuracy Logs:** Detailed logs of training loss and accuracy for both the training and test sets across every split are stored in the `train_history` folder.

#### 5.4 Plots of Training History

- **Visualization of Model Training:** Plots illustrating the training history, including loss and accuracy over epochs for each split, are saved in the `train_history_plots` folder.

# Table of Contents
- [Introducti](#introduction)
- [Example Header](#example-header)

## Introduction
Welcome to the document. Click here to jump to the [Example Header](#example-header) section.

## Example Header
This section provides detailed information.