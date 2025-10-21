import collections
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

class CategoryDistributionComparator:
    def __init__(self, graph):
        """
        graph: your Graph class instance with attribute `node_categories`:
            self.node_categories = {
                "descr_liv1": tensor(...),
                "descr_liv2": tensor(...),
                "descr_liv3": tensor(...),
                "descr_liv4": tensor(...)
            }
        """
        self.graph = graph
        self.category_levels = ["descr_liv1", "descr_liv2", "descr_liv3", "descr_liv4"]

    def _map_id_to_label(self, level: str, cat_id: int) -> Any:
        return self.graph.label_mappings[level]['id_to_label'].get(cat_id)

    def _get_category_distribution(self, nodes: List[int]) -> List[Dict[Any, float]]:
        level_counts = [collections.Counter() for _ in range(4)]
        total_counts = [0] * 4

        for node in nodes:
            for level_idx, level in enumerate(self.category_levels):
                categories_tensor = self.graph.node_categories[level]
                if node >= len(categories_tensor):
                    continue
                cat_id = int(categories_tensor[node].item())
                cat = self._map_id_to_label(level, cat_id)
                level_counts[level_idx][cat] += 1
                total_counts[level_idx] += 1

        distributions = []
        for level in range(4):
            if total_counts[level] == 0:
                distributions.append({})
            else:
                dist = {cat: count / total_counts[level] for cat, count in level_counts[level].items()}
                distributions.append(dist)
        return distributions

    def jensen_shannon_divergence(self, p: Dict[Any, float], q: Dict[Any, float]) -> float:
        all_keys = set(p.keys()).union(set(q.keys()))
        p_vec = np.array([p.get(k, 0.0) for k in all_keys])
        q_vec = np.array([q.get(k, 0.0) for k in all_keys])

        if p_vec.sum() > 0:
            p_vec = p_vec / p_vec.sum()
        if q_vec.sum() > 0:
            q_vec = q_vec / q_vec.sum()

        m = 0.5 * (p_vec + q_vec)
        def kl_divergence(a, b):
            mask = (a > 0)
            return np.sum(a[mask] * np.log(a[mask] / b[mask]))
        return 0.5 * kl_divergence(p_vec, m) + 0.5 * kl_divergence(q_vec, m)

    def evaluate_methods(self, **clusters_dicts) -> Dict[str, List[List[float]]]:
        """
        Accepts any number of cluster dicts as keyword arguments:
            e.g. evaluate_methods(pagerank=..., lightgcn=..., mymethod=...)
        Returns divergences per method:
            { method: [[lvl1 divergences], [lvl2], [lvl3], [lvl4]] }
        """
        results = {method: [[] for _ in range(4)] for method in clusters_dicts}
        # Union of all seeds from all methods
        all_seeds = set()
        for clusters in clusters_dicts.values():
            all_seeds.update(clusters.keys())

        for seed in all_seeds:
            # Convert seed to index if needed
            seed_idx = self.graph.product_to_index[seed] if isinstance(seed, str) else seed
            neighbors = list(self.graph.neighbors(seed_idx))
            neighbor_dist = self._get_category_distribution(neighbors)

            for method, clusters in clusters_dicts.items():
                cluster_nodes = clusters.get(seed, [])
                # Convert cluster_nodes to indices if needed
                cluster_indices = [self.graph.product_to_index[n] if isinstance(n, str) else n for n in cluster_nodes]
                cluster_dist = self._get_category_distribution(cluster_indices)
                for lvl in range(4):
                    div = self.jensen_shannon_divergence(neighbor_dist[lvl], cluster_dist[lvl])
                    results[method][lvl].append(div)

        return results

    def plot_results(self, divergences: Dict[str, List[List[float]]], mode: Optional[str] = None, range: Optional[str] = None):
        """
        Plot boxplots of divergences for each method.
        Each method will have 4 boxplots (one per category level).
        """
        num_methods = len(divergences)
        fig, axes = plt.subplots(1, num_methods, figsize=(5 * num_methods, 5), sharey=True)
        if num_methods == 1:
            axes = [axes]

        for ax, (method, levels) in zip(axes, divergences.items()):
            ax.boxplot(levels, labels=["lvl1", "lvl2", "lvl3", "lvl4"])
            ax.set_title(f"{method} divergences")
            ax.set_ylabel("Jensen-Shannon divergence")

        plt.tight_layout()
        plt.savefig(f"../Results/divergences_boxplots_{mode}_{range}.png")
        plt.show()
