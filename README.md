# Node Classification with Graph Convolutional Network (GCN): Cora Dataset

This project implements a Graph Convolutional Network (GCN) for node classification on the Cora dataset. The GCN model leverages node features and graph structure to classify nodes into different categories based on their content and citation relationships.

## Project Structure

### 1. Data Preparation
Data for the Cora dataset will be automatically downloaded if not present in a folder named "cora" in the current working directory. The dataset includes papers (nodes),embeddings (features), categories of papers (labels) and citation relationships (edges).

**Download Link:** [Cora Dataset](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz)

### 2. Model Architecture
The GCN model is defined with two graph convolutional layers. The first layer transforms input features to a hidden layer with 20 neurons, followed by a dropout layer for regularization, and a second graph convolutional layer that outputs the class scores.

### 3. Training Process
The Cora dataset is split into training and testing sets using stratified sampling to ensures that the label distribution is consistent between the train and test sets, which is crucial given the label imbalance in the dataset. Each split uses 80% of the data for training and 20% for testing.

During training:
- The model updates only using the training index in each batch.
- Loss and accuracy are reported every 100 epochs.
- Training history (loss and accuracy on train and test sets) will be saved in the `train_history` folder.
- Plots of training progress will be saved in the `train_history_plots` folder.

### 4. Running the Project

#### Configuration Setup
All model and training parameters are defined in the `configs/config.toml` file. This configuration file allows for easy adjustment of parameters.

#### With Conda
To run the project with conda, use the following commands:

```bash
conda create -n node_cora python=3.11
conda activate node_cora
pip install --no-cache-dir -r requirements.txt
chmod +x run.sh
./run.sh
```

### With Docker
```bash
docker-compose up
```
### 5. Results

After running the project, you can expect the following outputs, which help in evaluating the performance:

#### Prediction Outputs
- **Paper Categories Predictions:** The predictions for the categories of papers are saved in a `prediction.tsv` file. This file contains the predicted categories for each paper in the dataset.

#### Trained Models
- **Model Files:** The trained model for each data split is saved in the `models` folder. 

#### Training History
- **Loss and Accuracy Logs:** Detailed logs of training loss and accuracy for both the training and test sets across every split are stored in the `train_history` folder.

#### Plots of Training History
- **Visualization of Model Training:** Plots illustrating the training history, including loss and accuracy over epochs for each split, are saved in the `train_history_plots` folder. 


