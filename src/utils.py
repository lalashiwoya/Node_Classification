from typing import List, Tuple, Dict
from numpy.typing import NDArray
import numpy as np
import glob
from scipy.sparse import diags
from sklearn.model_selection import StratifiedShuffleSplit
import toml
import torch
import os
import csv

def read_file(fname: str) -> List[str]:
    data = []
    with open(fname, 'r') as f:
        data += f.read().splitlines()
    return data

def get_nodes_and_edges(folder: str) -> Tuple[List[str], List[str]]:
    all_nodes, all_edges = [], []
    
    for file in glob.glob(f"{folder}/*"):
        if ".content" in file:
            all_nodes = read_file(file)
        elif  ".cites" in file:
            all_edges = read_file(file)
    return all_nodes, all_edges

def parse_nodes(nodes: List[str]) -> Tuple[NDArray[np.str_], NDArray[np.float_], NDArray[np.str_]]:
    node_ids, node_embs, labels = [], [], []
    for data in nodes:
        elements = data.split('\t')
        node_ids.append(elements[0])
        node_embs.append(elements[1:-1])
        labels.append(elements[-1])
        
    return np.array(node_ids), np.array(node_embs, dtype=np.float32), np.array(labels)

def parse_edges(edges: List[str]) -> NDArray[np.str_]:
    all_edges = []
    for data in edges:
        element = data.split("\t")
        all_edges.append((element[0], element[1]))
    return np.array(all_edges)

def create_adj_matrix(edge_index: NDArray[np.int_]) -> NDArray[np.int_]:
    nodes_count = len(np.unique(edge_index))
    adj = np.zeros((nodes_count, nodes_count))
    for edge in edge_index:
        adj[edge[0], edge[1]] = 1
        adj[edge[1], edge[0]] = 1
    return adj

def normalize(mx: NDArray[np.int_]) -> NDArray[np.int_]:
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def get_splits(labels, n_splits, test_ratio, random_state=42):
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_ratio, random_state=random_state)
    sss.get_n_splits(range(len(labels)), labels)
    all_train_idx = []
    all_test_idx = []
    for i, (train_idx, test_idx) in enumerate(sss.split(range(len(labels)), labels)):
        all_train_idx.append(train_idx)
        all_test_idx.append(test_idx)
        
    return all_train_idx, all_test_idx
    
def get_config(path: str) -> Dict:
    with open(path, 'r') as toml_file:
        data = toml.load(toml_file)
        return data

def save_model(model, file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), file_path)
    print(f"Model successfully saved to {file_path}")

def load_model(model, file_path):
    model.load_state_dict(torch.load(file_path))
    print(f"Model successfully Loaded from {file_path}")
    return model

def write_train_history_to_file(train_loss, train_acc, test_loss, test_acc, file_path):
    with open(file_path, "a") as file:
        file.write(f"Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, "
                   f"Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}\n")


def write_preds_to_file(nodes, preds, file_path):
    with open(file_path, 'w', newline='') as tsvfile:
        tsv_writer = csv.writer(tsvfile, delimiter='\t')
        tsv_writer.writerow(['paper_id', 'class_label'])
        for node, pred in zip(nodes, preds):
            tsv_writer.writerow([str(node), pred])

