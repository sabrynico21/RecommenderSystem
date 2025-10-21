from graph_utils import *
import matplotlib.pyplot as plt
import numpy as np
import pickle
class PageRankNibbleTuner:
    def __init__(self, graph, beta=0.85, mode="unweighted", selected_nodes=None):
        self.graph = graph
        self.beta = beta
        self.epsilon = None
        self.mode = mode
        self.c = 0.002 if mode == "unweighted" else 0.00006
        self.selected_nodes = selected_nodes

    def tune_parameters(self):
        portion_non_connected = []
        clusters, times = calculate_clusters(self.graph, self.selected_nodes, self.mode, self.beta, self.c)
        with open("../data/clusters.pkl", "wb") as f:
                pickle.dump(clusters, f)
        for i, node in enumerate(self.selected_nodes):
            non_adjacent = extract_non_adjacent_nodes(self.graph, clusters[i], node)
            portion_non_connected.append(len(non_adjacent) / len(clusters[i]) if clusters[i] else 0)
        return portion_non_connected, times

    def plot_results(self, portion_non_connected):
        portion_non_connected = np.array(portion_non_connected)

        # Safety check: make sure it can be split into 3 groups
        if len(portion_non_connected) % 3 != 0:
            raise ValueError(
                f"Expected portion_non_connected to be divisible by 3, got {len(portion_non_connected)}"
            )

        groups = np.split(portion_non_connected, 3)  # 3 equal groups
        group_labels = ['Degree 10-100', 'Degree 100-1000', 'Degree 1000+']

        colors = ['#fc8d62','#8da0cb','#e78ac3']

        fig, ax = plt.subplots(figsize=(10,6))

        box = ax.boxplot(
            groups,
            labels=group_labels,
            patch_artist=True,
            showmeans=True,  # highlight the mean
            meanline=True
        )

        # Apply colors
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        # Style improvements
        ax.set_ylabel('Proportion of Non-Connected Nodes', fontsize=12)
        ax.set_title('Non-Connected Node Proportion by Seed Degree Range', fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()


        