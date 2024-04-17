from src.utils import get_config
import numpy as np
from src.dataset import CoraData
from src.models import GCN
from src.utils import load_model
from src.predict import predict
import torch
from src.utils import write_preds_to_file
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Make predictions using parameters 
                                         specified in a configuration TOML file.""")  
        
    parser.add_argument("--config_path", type=str, help="Location of config file", required=True)
    args = parser.parse_args()
    config = get_config(args.config_path)
    
            
    cora_folder_path = config['data']['cora_folder_path']
    n_hidden_neurons = config['model']['n_hidden_neurons']
    dropout = config['model']['dropout']
    model_save_dir = config['model']['model_save_dir']
    model_save_name = config['model']['model_save_name']
    pred_save_name = config['prediction']['pred_save_name']




    dataset = CoraData(cora_folder_path)

    nodes = dataset.nodes
    labels = dataset.labels
    adj = dataset.adj
    features = dataset.features
    edges = dataset.edges
    labels_to_idx  = dataset.labels_to_idx
    nodes_to_idx = dataset.nodes_to_idx

    n_labels = len(np.unique(labels))
    n_features = features.shape[1]

    model = GCN(nfeat=n_features,
                    nhid=n_hidden_neurons,  
                    nclass=n_labels,
                    dropout=dropout)  

    model_path = f"{model_save_dir}/{model_save_name}"
    model = load_model(model, model_path)

    device = "cpu"

    adj = torch.FloatTensor(adj).to(device)
    features = torch.FloatTensor(features).to(device)
    labels = torch.LongTensor(labels).to(device)
    preds  = predict(features, adj, labels, range(len(labels)), model, device)

    idx_to_labels = {v:k for k, v in labels_to_idx.items()}
    pred_labels = np.vectorize(lambda x: idx_to_labels[x])(preds)

    pred_save_path = pred_save_name + ".tsv"
    write_preds_to_file(nodes, pred_labels, pred_save_path)

