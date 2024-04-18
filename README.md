# Node Classification with Graph Convolutional Network (GCN): Cora Dataset

This project implements a Graph Convolutional Network (GCN) for node classification on the Cora dataset. The GCN model leverages node features (embeddings) and graph structure (adjacency matrix) to classify nodes into different categories based on their content and citation relationships.

## Project Structure

### 1. Data Preparation
Data for the [Cora dataset](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz) will be automatically downloaded if not present in a folder named "cora" in the current working directory. The dataset includes papers (nodes), embeddings (features), categories of papers (labels), and citation relationships (edges).

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
- Loss and accuracy are reported every 100 epochs.
- Training history (loss and accuracy on train and test sets) is saved in the `train_history` folder.
- Plots of training progress are saved in the `train_history_plots` folder.

### 4. Running the Project (Training and Predicting)

#### Configuration Setup
All model and training parameters are defined in the `configs/config.toml` file. This configuration file allows for easy adjustment of parameters.

#### Run with Conda
To run the project with Conda, use the following commands:

```bash
conda create -n node_cora python=3.11
conda activate node_cora
pip install --no-cache-dir -r requirements.txt
chmod +x run.sh
./run.sh
```
### Run with Docker

```bash
docker-compose up --build
```

### 5. Results

After running the project, you can expect the following outputs, which help in evaluating the performance:

#### 5.1 Prediction Outputs
- **Paper Categories Predictions:** The predictions for the categories of papers are saved in a `prediction.tsv` file. This file contains the predicted categories for each paper in the dataset. If you only need the prediction file, you can use the trained model located in `models/model_split_1` to make predictions. Here are the instructions:

##### With Conda
```bash
conda create -n node_cora python=3.11
conda activate node_cora
pip install --no-cache-dir -r requirements.txt
chmod +x run.predict.sh
./run.predict.sh
```
##### With Docker

```bash
docker-compose -f docker-compose.predict.yaml up --build
```
#### 5.2 Trained Models

- **Model Files:** The trained models for each data split are saved in the `models` folder.

#### 5.3 Training History

- **Loss and Accuracy Logs:** Detailed logs of training loss and accuracy for both the training and test sets across every split are stored in the `train_history` folder.

#### 5.4 Plots of Training History

- **Visualization of Model Training:** Plots illustrating the training history, including loss and accuracy over epochs for each split, are saved in the `train_history_plots` folder.

