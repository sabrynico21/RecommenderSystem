import os
import pickle
import random
from collections import defaultdict
import numpy as np
import networkx as nx
import math
import re
from collections import defaultdict
import torch 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Graph:
    def __init__(self, node_labels=None):
        self.graph = nx.Graph()
        self.product_to_index = dict()
        self.index_to_product = dict()
        self.t_min = 0
        self.t_max = float('inf')
        self.deg = defaultdict(int)
        if node_labels is not None:
            self.node_categories = {}
            for level in ["descr_liv1", "descr_liv2", "descr_liv3", "descr_liv4"]:
                if level in node_labels.columns:
                    self.node_categories[level] = torch.tensor(
                        node_labels[level].values, 
                        dtype=torch.long,
                        device=device
                    )
            # if "descr_forn" in node_labels.columns:
            #     self.node_supplier = torch.tensor(
            #         node_labels["descr_forn"].values,
            #         dtype=torch.long,
            #         device=device
            #     )

            # if "descr_rep" in node_labels.columns:
            #     self.node_department = torch.tensor(
            #         node_labels["descr_rep"].values,
            #         dtype=torch.long,
            #         device=device
            #     )

    def __getattr__(self, attr):
        if attr == "graph":
            return super().__getattribute__("graph")
        return getattr(self.graph, attr)
    
    def add_edge(self, node1, node2, weight=None):
        if weight is not None:
            self.graph.add_edge(node1, node2, weight=weight)
        else:
            self.graph.add_edge(node1, node2)

    def set_t_min(self, t_min):
        self.t_min = t_min
    
    def set_t_max(self, t_max):
        self.t_max = int(t_max) if not math.isinf(t_max) else t_max

    def get_graph(self):
        return self.graph
    
    def get_subgraph(self, cluster):
        return self.graph.subgraph(cluster)
    
    def get_product_to_index(self, x=None):
        if x is None:
            return self.product_to_index
        else:
            return self.product_to_index[x]
    
    def get_index_to_product(self, x=None):
        if x is None:
            return self.index_to_product
        else:
            return self.index_to_product[x]
        
    def get_neighbors(self, node):
        if node in self.graph:
            return list(self.graph.neighbors(node))
        return []
    
    def get_weight(self, u, v):
        if self.graph.has_node(u) and self.graph.has_node(v) and self.graph.has_edge(u, v):
            return self.graph[u][v].get('weight', 1)
        return 0
        
    def get_degree(self, nodes=None, weight=None):
        if nodes is None:
            if weight is None:
                return self.graph.degree()
            else:
                return self.graph.degree(weight=weight)
        else:
            if weight is None:
                return self.graph.degree(nodes)
            else:
                return self.graph.degree(nodes, weight=weight)

    def mean_degree(self, nodes):
        degrees = [self.deg[self.index_to_product[n]] for n in nodes]
        return np.mean(degrees) if degrees else 0
    
    def copy(self):
        new_graph = Graph()
        new_graph.graph = self.graph.copy()
        new_graph.product_to_index = self.product_to_index.copy()
        new_graph.index_to_product = self.index_to_product.copy()
        new_graph.t_min = self.t_min
        new_graph.t_max = self.t_max 
        return new_graph

    def create_graph(self, edge_weights, t_min=0, t_max=float('inf')):
        self.set_t_min(t_min)
        self.set_t_max(t_max)
        current_index = 0
        for edge, weight in edge_weights.items():
            self.deg[edge[0]] += 1
            self.deg[edge[1]] += 1
            if weight >= self.t_min and weight <= self.t_max: 
                product_i, product_j = edge

                for p in [product_i, product_j]:
                    if p not in self.product_to_index:
                        self.product_to_index[p] = current_index
                        self.index_to_product[current_index] = p
                        current_index += 1

                product_i_index = self.product_to_index[product_i]
                product_j_index = self.product_to_index[product_j]

                self.add_edge(product_i_index, product_j_index, weight=weight)   
        return

    @classmethod
    def load_graph(cls, file_path, t_min, t_max):
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                graph_data = pickle.load(f)
            print("Graph loaded successfully.")
            instance = cls()  
            instance.graph = graph_data  
            instance.set_t_min(t_min)
            instance.set_t_max(t_max)
            return instance
        else:
            print(f"File {file_path} does not exist.")
            return None
        
    def load_dicts(self, dict_path):
        if os.path.exists(dict_path):
            with open(dict_path, 'rb') as f:
                data = pickle.load(f)
            self.product_to_index = data['id_to_index']
            self.index_to_product = data['index_to_id']
        else:
            print(f"File {dict_path} does not exist.")
            return None
    
    def subtract_edgeweights(self, weights_to_remove):
        new_graph = Graph()
        new_graph.graph.add_nodes_from(self.graph.nodes(data=True))
        
        for u, v, data in self.graph.edges(data=True):
            weight_to_remove = weights_to_remove.get((u, v), 0)
            new_weight = data.get('weight', 0) - weight_to_remove
            if new_weight >= self.t_min:
                new_graph.add_edge(u, v, weight=new_weight)
        
        return new_graph
    
    def select_random_edges(self, weight_min, weight_max, num):
        valid_edges = [(self.index_to_product[u], self.index_to_product[v], w) for u, v, w in self.graph.edges(data="weight") if weight_min <= w < weight_max] #Choose a specific pair of products that were sold together
        local_random = random.Random(42)
        random_edges = list(local_random.sample(valid_edges, num) if valid_edges else None)
        return random_edges

    def new_graph_removing_receipts(self, client, random_edge):
        query = f"SELECT * FROM grouped_products WHERE match(products, '(^|\\s){random_edge[0]}(\\s|$)') AND match(products, '(^|\\s){random_edge[1]}(\\s|$)');"
        result = client.query(query)
        weights_to_remove = defaultdict(int)
        for row in result.result_rows:
            products = list(set(row[1].split(' ')))
            for i in range(len(products)):
                for j in range(i + 1, len(products)):
                    if products[i] not in self.product_to_index or products[j] not in self.product_to_index:
                        continue
                    edge = (self.product_to_index[products[i]], self.product_to_index[products[j]])
                    weights_to_remove[edge] += 1
        new_graph = self.subtract_edgeweights(weights_to_remove)
        return new_graph

    def new_graph_removing_receipts_from_df(self, df, random_edge):
        # Create regex patterns to match full words (tokens)
        p1, p2 = random_edge[0], random_edge[1]
        pattern1 = re.compile(rf'\b{re.escape(p1)}\b')
        pattern2 = re.compile(rf'\b{re.escape(p2)}\b')

        # Filter rows where both products appear in the 'products' column
        #filtered_df = df[df['products'].apply(lambda x: bool(pattern1.search(x)) and bool(pattern2.search(x)))]
        filtered_df = df[df['products'].apply(lambda x: bool(pattern1.search(str(x))) and bool(pattern2.search(str(x))))]

        weights_to_remove = defaultdict(int)

        for _, row in filtered_df.iterrows():
            products = list(set(row['products'].split(' '))) 
            for i in range(len(products)):
                for j in range(i + 1, len(products)):
                    if products[i] not in self.product_to_index or products[j] not in self.product_to_index:
                        continue
                    edge = (self.product_to_index[products[i]], self.product_to_index[products[j]])
                    weights_to_remove[edge] += 1

        new_graph = self.subtract_edgeweights(weights_to_remove)
        return new_graph

    def remove_random_edges(self, percentage):
        graph_copy = self.graph.copy()
        num_edges_to_remove = int(percentage * self.graph.number_of_edges() / 100)
        edges = list(self.graph.edges())   
        edges_to_remove = random.sample(edges, num_edges_to_remove)
        graph_copy.remove_edges_from(edges_to_remove)    
        return graph_copy