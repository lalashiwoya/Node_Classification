from dataset import CoraData
import numpy as np
import torch
from models import GCN
from utils import get_splits
import torch.optim as optim
from train import train
 
torch.manual_seed(42)

cora_folder_path = "cora"

dataset = CoraData(cora_folder_path)

nodes = dataset.nodes
labels = dataset.labels
adj = dataset.adj
features = dataset.features
edges = dataset.edges

n_labels = len(np.unique(labels))
n_features = features.shape[1]

n_hidden_neurons= 20
dropout = 0.5
n_splits = 10
test_ratio = 0.2

model = GCN(nfeat=n_features,
            nhid=n_hidden_neurons,  
            nclass=n_labels,
            dropout=dropout)  

device = "cpu"

all_train_idx, all_test_idx = get_splits(labels, n_splits, test_ratio)

adj = torch.FloatTensor(adj).to(device)
features = torch.FloatTensor(features).to(device)
labels = torch.LongTensor(labels).to(device)

lr = 0.001
weight_decay = 5e-4
num_epochs = 500
batch_size = 1000

loss_func =torch.nn.CrossEntropyLoss()

for i in range(n_splits):
    print("*"*20)
    print(f"Start of Split {i+1}")
    model = GCN(nfeat=n_features,
            nhid=20,  
            nclass=n_labels,
            dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(),
                       lr=lr, 
                       weight_decay=weight_decay)
    train_idx = torch.LongTensor(all_train_idx[i]).to(device)
    test_idx = torch.LongTensor(all_test_idx[i]).to(device)
    
    train(features, adj, labels, train_idx, test_idx, model, optimizer, loss_func, num_epochs, batch_size, device)
    
    print(f"End of Split {i+1}")
    print("*"*20)
    print("\n")






 



