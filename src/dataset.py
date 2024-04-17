from .utils import (get_nodes_and_edges, 
                   parse_edges, 
                   parse_nodes, 
                   normalize, 
                   create_adj_matrix)
from typing import Any, Dict
from numpy.typing import NDArray
import numpy as np

class CoraData:
    def __init__(self, folder_name: str):
        self.folder_name = folder_name
        self.nodes, self.labels, self. adj, self.features, self.edges = self._get_node_label_adj_embedding()
    
    def _categories_to_idx(self, x: NDArray[Any], mapping: Dict) -> NDArray[np.int_]:
        idx = np.vectorize(lambda key: mapping[key])(x)
        return idx
        
    def _get_node_label_adj_embedding(self):
        all_nodes, all_edges = get_nodes_and_edges(self.folder_name)
        node_ids, node_embs, node_labels = parse_nodes(all_nodes)
        edges = parse_edges(all_edges)
        labels_to_idx = {x: idx for idx, x in enumerate(np.unique(node_labels))}
        nodes_to_idx = {x: idx for idx, x in enumerate(node_ids)}
        
        nodes_idx = self._categories_to_idx(node_ids, nodes_to_idx)
        label_idx = self._categories_to_idx(node_labels, labels_to_idx)
        edges_idx = self._categories_to_idx(edges, nodes_to_idx)
        
        adj = create_adj_matrix(edges_idx)
        adj = normalize(adj)
        
        node_embs = normalize(node_embs)
        
        return nodes_idx, label_idx, adj, node_embs, edges_idx

