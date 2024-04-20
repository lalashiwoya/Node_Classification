from src.dataset import CoraData
import numpy as np
import torch
from src.models import GCN
from src.utils import get_splits, get_config, load_model, write_train_history_to_file, plot_all_training_on_all_splits
import torch.optim as optim
from src.train import train
import os
import argparse
from src.predict import predict
torch.manual_seed(42)


if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="""Train a model using parameters 
                                         specified in a configuration TOML file.""")  
        
        parser.add_argument("--config_path", type=str, help="Location of config file", required=True)
        parser.add_argument("--cv_method", type=str, help="Methods of splitting dataset", default="stratified")
        args = parser.parse_args()
        config = get_config(args.config_path)
        cv_method = args.cv_method
        
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
        train_history_dir= config['train']["train_history_dir"]
        print_history = config['train']["print_history"]
        plot_history = config['train']["plot_history"]
        train_history_plot_dir = config['train']["train_history_plot_dir"]
        
        if not os.path.exists(train_history_dir):
                os.makedirs(train_history_dir, exist_ok=True)



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

        all_train_idx, all_test_idx = get_splits(labels, n_splits, test_ratio, cv=cv_method)

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
                
                if print_history:
                        train_history_path = f"{train_history_dir}/split_{i+1}.txt"
                        write_train_history_to_file(None, None, None, None, train_history_path, mode='w')
                train(features, adj, labels, train_idx, test_idx, model, optimizer, loss_func, num_epochs, batch_size, device, model_path, 
                        train_history_path, print_history)
                
                #     model = load_model(model, model_path)
                #     model(features, adj)
                # print(predict(features, adj, labels, test_idx, model, device))
                print(f"End of Split {i+1}")
                print("*"*20)
                print("\n")
        if plot_history:
                plot_all_training_on_all_splits(train_history_dir, train_history_plot_dir)






 



