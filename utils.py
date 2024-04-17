from typing import List, Tuple
from numpy.typing import NDArray
import numpy as np
import glob
from scipy.sparse import diags
from sklearn.model_selection import StratifiedShuffleSplit


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

def parse_nodes(nodes: List[str]) -> Tuple[NDArray[np.int_], NDArray[np.float_], NDArray[np.str_]]:
    node_ids, node_embs, labels = [], [], []
    for data in nodes:
        elements = data.split('\t')
        node_ids.append(int(elements[0]))
        node_embs.append(elements[1:-1])
        labels.append(elements[-1])
        
    return np.array(node_ids), np.array(node_embs, dtype=np.float32), np.array(labels)

def parse_edges(edges: List[str]) -> NDArray[np.int_]:
    all_edges = []
    for data in edges:
        element = data.split("\t")
        all_edges.append((int(element[0]), int(element[1])))
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
        
        