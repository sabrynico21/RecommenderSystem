from graph_utils import *
import pickle

def compare_node_categories(graph):
        """
        For each node and each category level, computes the proportion of neighbors
        that share the same category value as the node.
        Returns: {node: [proportion_level1, proportion_level2, proportion_level3, proportion_level4]}
        """
        results = {}
        num_levels = 4
        for node in graph.nodes():
            proportions = []
            for level in graph.levels_names:
                node_cat = graph.node_categories[level][node].item()
                neighbors = list(graph.neighbors(node))
                if not neighbors:
                    proportions.append(None)  # or 0.0 if you prefer
                    continue
                same_cat_count = sum(
                    graph.node_categories[level][nbr].item() == node_cat for nbr in neighbors
                )
                proportion = same_cat_count / len(neighbors)
                proportions.append(proportion)
            results[node] = proportions
        return results

import matplotlib.pyplot as plt
import numpy as np

def plot_category_similarity_boxplots(results, graph_type):
    """
    Displays 4 box plots, one for each category level, showing the distribution
    of the proportion of neighbors sharing the same category value.
    """
    # Collect proportions for each level
    num_levels = 4
    level_names = ["descr_liv1", "descr_liv2", "descr_liv3", "descr_liv4"]
    data = [[] for _ in range(num_levels)]
    for node, proportions in results.items():
        for i, prop in enumerate(proportions):
            if prop is not None:
                data[i].append(prop)

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=level_names, patch_artist=True)
    plt.ylabel("Proportion of neighbors with same category")
    plt.title("Category Similarity Across Levels")
    plt.grid(axis='y')
    plt.savefig(f"../Results/{graph_type}_category_similarity_boxplots.png")
    plt.show()

class LightGCNTuner:
    def __init__(self, model, train_graph, data):
        self.model = model
        self.train_graph = train_graph
        self.data = data

    def pruning_tuning(self):
        self.model.eval()
        node_embeddings = self.model.get_embeddings(self.data.edge_index, edge_weight=self.data.edge_attr)
        print("Embeddings shape:", node_embeddings.shape) 
        G_pruned = prune_graph(self.train_graph, node_embeddings, sim_threshold=0.8)
        print("G pruned - Number of nodes:", G_pruned.number_of_nodes())
        print("G pruned - Number of edges:", G_pruned.number_of_edges())

        with open(f"../data/train_pruned.pkl", "wb") as f:
            pickle.dump(G_pruned, f)
        results = compare_node_categories(G_pruned)
        return results
    

    
    