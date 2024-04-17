from dataset import CoraData
import numpy as np
import torch
from models import GCN
from utils import get_splits, get_config, load_model
import torch.optim as optim
from train import train
import os
import argparse
torch.manual_seed(42)


if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="""Train a model using parameters 
                                         specified in a configuration TOML file.""")  
        
        parser.add_argument("--config_path", type=str, help="Location of config file")
        args = parser.parse_args()
        config = get_config(args.config_path)
        
        cora_folder_path = config['data']['cora_folder_path']
        n_splits = config['split']['n_splits']
        test_ratio = config['split']['test_ratio']
        n_hidden_neurons = config['model']['n_hidden_neurons']
        dropout = config['model']['dropout']
        model_save_dir = config['model']['model_save_dir']
        model_save_name = config['model']['model_save_name']
        lr = config['train']['lr']
        weight_decay = config['train']['weight_decay']
        num_epochs = config['train']['num_epochs']
        batch_size = config['train']['batch_size']
        train_history_name = config['train']["train_history_name"]



        dataset = CoraData(cora_folder_path)

        nodes = dataset.nodes
        labels = dataset.labels
        adj = dataset.adj
        features = dataset.features
        edges = dataset.edges

        n_labels = len(np.unique(labels))
        n_features = features.shape[1]


        model = GCN(nfeat=n_features,
                nhid=n_hidden_neurons,  
                nclass=n_labels,
                dropout=dropout)  

        device = "cpu"

        all_train_idx, all_test_idx = get_splits(labels, n_splits, test_ratio)

        adj = torch.FloatTensor(adj).to(device)
        features = torch.FloatTensor(features).to(device)
        labels = torch.LongTensor(labels).to(device)
        loss_func =torch.nn.CrossEntropyLoss()



        for i in range(n_splits):
                print("*"*20)
                print(f"Start of Split {i+1}")
                
                model = GCN(nfeat=n_features,
                        nhid=n_hidden_neurons,  
                        nclass=n_labels,
                        dropout=dropout).to(device)
                optimizer = optim.Adam(model.parameters(),
                                lr=lr, 
                                weight_decay=weight_decay)
                train_idx = torch.LongTensor(all_train_idx[i]).to(device)
                test_idx = torch.LongTensor(all_test_idx[i]).to(device)
                
                model_path = os.path.join(model_save_dir, f"{model_save_name}_split_{i+1}")
                
                train_history_path = train_history_name + f"_split_{i+1}" + ".txt"
                train(features, adj, labels, train_idx, test_idx, model, optimizer, loss_func, num_epochs, batch_size, device, model_path, 
                        train_history_path)
                
                #     model = load_model(model, model_path)
                #     model(features, adj)
                
                print(f"End of Split {i+1}")
                print("*"*20)
                print("\n")






 



