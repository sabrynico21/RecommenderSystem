import os
import argparse
import pickle
from collections import defaultdict
from dotenv import load_dotenv
#import clickhouse_connect
from graph import Graph
from graph_utils import *
from LightGCN import *
import torch
import ast
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def calculate_labels(train_graph, test_graph, selected_nodes):
    train_neighbors = defaultdict(set)
    test_neighbors = defaultdict(set)
    ground_truth = defaultdict(set)
    selected_nodes = selected_nodes[:20]
    for node in selected_nodes:
        t_n =train_graph.get_neighbors(train_graph.get_product_to_index(node))
        train_neighbors[node] = set([train_graph.get_index_to_product(n) for n in t_n])
        t_n = test_graph.get_neighbors(test_graph.get_product_to_index(node))
        test_neighbors[node] = set([test_graph.get_index_to_product(n) for n in t_n])
        ground_truth[node] = set(test_neighbors[node] - train_neighbors[node])
    return ground_truth

def compute_set_metrics(ground_truth, predictions):
    precisions, recalls, f1s = [], [], []
    for node in ground_truth:
        gt = ground_truth[node]
        pred = predictions.get(node, set())
        tp = len(gt & pred)
        fp = len(pred - gt)
        fn = len(gt - pred)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f1 = sum(f1s) / len(f1s)
    return avg_precision, avg_recall, avg_f1

def filter_clusters(G_pruned, non_adjacent, selected_node, node_embeddings):
    ref_idx = G_pruned.get_product_to_index(selected_node)
    ref_emb = node_embeddings[ref_idx]
    filtered = []
    for idx in non_adjacent:
        sim = torch.nn.functional.cosine_similarity(ref_emb, node_embeddings[idx], dim=0)
        if sim > 0.7:
            filtered.append(idx)
    return filtered 
    

def get_similar_products(graph, node_embeddings, product_idx, threshold=0.8):
    product_embedding = node_embeddings[product_idx]
    similarities = torch.nn.functional.cosine_similarity(product_embedding.unsqueeze(0), node_embeddings, dim=1)
    
    # Get indices of nodes connected to product_idx (including itself)
    connected = set(graph.get_neighbors(product_idx))
    connected.add(product_idx)  # Exclude self as well

    # Find indices with similarity > threshold and not connected
    similar_indices = [idx for idx, sim in enumerate(similarities) if sim > threshold and idx not in connected]
    similar_scores = similarities[similar_indices]

    return similar_indices, similar_scores

def get_edge_weights_from_nx(G):
    return {(u, v): d.get('weight', 1) for u, v, d in G.edges(data=True)}

def converting_metadata(metadata):
    df = pd.DataFrame.from_dict(metadata, orient="index")
    print(df.head())
    encoders = {}
    for level in ["descr_liv1", "descr_liv2", "descr_liv3", "descr_liv4", "descr_rep", "descr_forn"]:
        le = LabelEncoder()
        df[level] = le.fit_transform(df[level])
        encoders[level] = le
    print(df.head())
    print(df.columns)
    with open("../data/metadata_labels.pkl", "wb") as f:
        pickle.dump(df, f)

def simple_align_labels(node_labels_df, product_to_node_map):
    """
    Simple alignment - removes nodes that don't exist in the map
    """
    print('node labels len')
    rows, columns = node_labels_df.shape
    print(f"Rows: {rows}, Columns: {columns}")
    print('product to node map len', len(product_to_node_map.keys()))
    # Get only the product IDs that exist in both the DataFrame and the map
    common_products = set(node_labels_df.index) & set(product_to_node_map.keys())
    
    if not common_products:
        raise ValueError("No common products found between node_labels_df and product_to_node_map")
    
    # Filter DataFrame and map node IDs
    aligned_df = node_labels_df.loc[list(common_products)].copy()
    aligned_df.index = [product_to_node_map[product_id] for product_id in aligned_df.index]
    
    # Sort by node ID
    aligned_df = aligned_df.sort_index()
    print('node labels len')
    rows, columns = aligned_df.shape
    print(f"Rows: {rows}, Columns: {columns}")
    return aligned_df

def main():

    parser = argparse.ArgumentParser(description='Insert thresholds for the graph cluster algorithm')
    parser.add_argument('--load', default= "True", help='Specify if load the graph or create a new one')
    parser.add_argument('--mode', default="weighted", choices=["weighted","unweighted"], help='Specify whether to use weighted or unweighted graph')
    parser.add_argument('--task', default="comparison", choices=["training", "pruning","comparison", "prova"], help='Specify the task to perform')
    parser.add_argument('--random', default="False", help='Specify if use random selection of nodes/edges for the selected task')
    args = parser.parse_args()
    load_dotenv()

    if args.task == "training":
        if (os.path.exists(f"../data/train_edge_weights.pkl") and os.path.exists(f"../data/val_edge_weights.pkl")):
            with open(f"../data/train_edge_weights.pkl", "rb") as f:
                train_edge_weights = pickle.load(f)
            with open(f"../data/val_edge_weights.pkl", "rb") as f:
                val_edge_weights = pickle.load(f)
        else:
            # client = clickhouse_connect.get_client(
            #         host=os.getenv('CLICKHOUSE_HOST'),
            #         port=int(os.getenv('CLICKHOUSE_PORT')),
            #         username=os.getenv('CLICKHOUSE_USER'),
            #         password=os.getenv('CLICKHOUSE_PASSWORD'),
            #         database=os.getenv('CLICKHOUSE_DATABASE')
            #         )
            table_name = 'dati_scontrini'
            train_edge_weights, test_edge_weights = calculate_edge_weights(client, table_name, "split")
            with open(f"../data/train_edge_weights.pkl", "wb") as f:
                pickle.dump(train_edge_weights, f)
            with open(f"../data/test_edge_weights.pkl", "wb") as f:
                pickle.dump(test_edge_weights, f)

        train_graph = Graph()
        val_graph = Graph()
        train_graph.create_graph(train_edge_weights, int(args.t_min), float(args.t_max)) 
        val_graph.create_graph(val_edge_weights, int(args.t_min), float(args.t_max))

        with open ("../data/metadata_labels.pkl", "rb") as f:
            metadata = pickle.load(f)
        
        metadata = simple_align_labels(metadata, train_graph.product_to_index)
        node_embeddings = train_lightgcn(train_graph, metadata, val_graph)
        print("Embeddings shape:", node_embeddings.shape)  # [num_nodes, 64]
    
    elif args.task == "pruning":
        if (os.path.exists(f"../data/train_edge_weights.pkl") and os.path.exists(f"../data/test_edge_weights.pkl")):
            with open(f"../data/train_edge_weights.pkl", "rb") as f:
                train_edge_weights = pickle.load(f)
        train_graph = Graph()
        train_graph.create_graph(train_edge_weights, int(args.t_min), float(args.t_max)) 
        data = create_pyg_data_from_networkx(train_graph, weight_attr='weight')
        model = LightGCN(num_nodes=data.num_nodes, embedding_dim=64, num_layers=3)

        # Load the saved weights
        model.load_state_dict(torch.load("../LightGCN/lightgcn_model_099.pth"))
        model.eval()
        node_embeddings = model.get_embeddings(data.edge_index, edge_weight=data.edge_attr)
        print("Embeddings shape:", node_embeddings.shape)  # [num_nodes, 64]

        G_pruned = prune_graph(train_graph, node_embeddings, weight_threshold=50, sim_threshold=0.8)
        print("G pruned - Number of nodes:", G_pruned.number_of_nodes())
        print("G pruned - Number of edges:", G_pruned.number_of_edges())
        with open(f"../data/G_pruned_p.pkl", "wb") as f:
            pickle.dump(G_pruned, f)
    elif args.task == "comparison":
        with open("../data/selected_nodes.txt") as f:
            selected_nodes = ast.literal_eval(f.read())
        selected_nodes = selected_nodes[:20]

        with open("../data/G_pruned_p.pkl", "rb") as f:
            G_pruned = pickle.load(f)
        print("Train Graph Pruned - Number of nodes:", G_pruned.number_of_nodes())
        print("Train Graph Pruned - Number of edges:", G_pruned.number_of_edges())
        #G_pruned_edge_weights = get_edge_weights_from_nx(G_pruned.graph)
        #fit_powerlaw_on_edge_distribution(G_pruned_edge_weights)
        #fit_powerlaw_on_degree_distribution(G_pruned)
        if (os.path.exists(f"../data/ground_truth.pkl")):
            with open("../data/ground_truth.pkl", "rb") as f:
                ground_truth = pickle.load(f)
        else:
            if (os.path.exists(f"../data/train_edge_weights.pkl") and os.path.exists(f"../data/test_edge_weights.pkl")):
                with open(f"../data/train_edge_weights.pkl", "rb") as f:
                    train_edge_weights = pickle.load(f)
                with open(f"../data/test_edge_weights.pkl", "rb") as f:
                    test_edge_weights = pickle.load(f)

            train_graph = Graph()
            test_graph = Graph()
            train_graph.create_graph(train_edge_weights, int(args.t_min), float(args.t_max)) 
            test_graph.create_graph(test_edge_weights, int(args.t_min), float(args.t_max))
        
            train_neighbors = defaultdict(set)
            test_neighbors = defaultdict(set)
            ground_truth = defaultdict(set)
            
            for node in selected_nodes:
                t_n =train_graph.get_neighbors(train_graph.get_product_to_index(node))
                train_neighbors[node] = set([train_graph.get_index_to_product(n) for n in t_n])

                node_idx = test_graph.get_product_to_index(node)
                t_n = test_graph.get_neighbors(node_idx)
                filtered_neighbors = [
                    test_graph.get_index_to_product(n)
                    for n in t_n
                    #if test_graph.get_weight(node_idx, n) > 1
                ]
                test_neighbors[node] = set(filtered_neighbors)
                candidates = list(set(test_neighbors[node] - train_neighbors[node]))
                gt = [c for c in candidates if c in train_graph.product_to_index]
                ground_truth[node] = set(gt)
            
            with open("../data/ground_truth.pkl", "wb") as f:
                pickle.dump(ground_truth, f)
            print("ottenute etichette e salvate")

        data = create_pyg_data_from_networkx(train_graph, weight_attr='weight')
        model = LightGCN(num_nodes=data.num_nodes, embedding_dim=64, num_layers=3)

        # Load the saved weights
        model.load_state_dict(torch.load("../LightGCN/lightgcn_model_099.pth"))
        model.eval()
        node_embeddings = model.get_embeddings(data.edge_index, edge_weight=data.edge_attr)
        print("Embeddings shape:", node_embeddings.shape) 

        selected_nodes_indices = [G_pruned.get_product_to_index(node) for node in selected_nodes]
        embeddings_predictions = defaultdict(set)
        for i,node_idx in enumerate(selected_nodes_indices):
            similar_indices, similar_scores = get_similar_products(train_graph, node_embeddings, node_idx)
            embeddings_predictions[selected_nodes[i]] = set([G_pruned.get_index_to_product(x) for x in similar_indices])
        #similar_indices = similar_indices.tolist()
        #print(f"Products similar to product 0: {[train_graph.get_index_to_product(x) for x in similar_indices]}")
        #intersections = {key: ground_truth[key] & embeddings_predictions.get(key, set()) for key in set(ground_truth) | set(embeddings_predictions)}
        #non_empty_count = sum(1 for intersection_set in intersections.values() if intersection_set)
        #print(f"Number of nodes with non-empty intersection: {non_empty_count} out of {len(selected_nodes)}")

        clusters, _ = calculate_clusters(G_pruned, selected_nodes_indices, args.mode)
        #clusters_filtered = []
        #tensor_data = torch.tensor(clusters, dtype=torch.float32)
        #torch.save(tensor_data, "../data/pagerank_features.pt")
        #exit(0)
        #top_15_clusters = [[x for x in clusters[i][1:16]] for i in range(len(clusters))]
        clusters_predictions = defaultdict(set)
        #clusters_predictions_filtered = defaultdict(set)
        for i, node in enumerate(selected_nodes):
            non_adjacent = extract_non_adjacent_nodes(train_graph, clusters[i], G_pruned.get_product_to_index(selected_nodes[i]))
            #print("len non adiacenti:", len(non_adjacent))
            if len(non_adjacent) > 0:
               clusters[i] = non_adjacent
            #    filtered = filter_clusters(G_pruned, non_adjacent, selected_nodes[i], node_embeddings)
            # else:
            #    filtered = []
            # clusters_filtered.append(filtered)
            clusters_predictions[node] = set([G_pruned.get_index_to_product(x) for x in clusters[i]])
            #clusters_predictions_filtered[node] = set([G_pruned.get_index_to_product(x) for x in clusters_filtered[i]])
            #print("ground truth", ground_truth[node])
            #print("top 15", top_15_clusters[node])
        #intersections_clusters = {key: ground_truth[key] & clusters_predictions.get(key, set()) for key in set(ground_truth) | set(clusters_predictions)}
        #intersections_clusters_filtered = {key: ground_truth[key] & clusters_predictions_filtered.get(key, set()) for key in set(ground_truth) | set(clusters_predictions_filtered)}
        emb_intersections = []
        clu_intersections = []
        for key in selected_nodes:
            gt = ground_truth.get(key, set())
            emb_pred = embeddings_predictions.get(key, set())
            clu_pred = clusters_predictions.get(key, set())
            intersection = gt & emb_pred
            emb_intersections.append(len(intersection))
            intersection = gt & clu_pred
            clu_intersections.append(len(intersection))
            #denom = min(len(pred), len(gt))
            #ratio = len(intersection) / denom if denom > 0 else 0.0

        union_predictions = defaultdict(set)
        union_intersections = []

        for key in selected_nodes:
            emb_pred = embeddings_predictions.get(key, set())
            clu_pred = clusters_predictions.get(key, set())
            union_pred = emb_pred | clu_pred  # Union of both sets
            union_predictions[key] = union_pred

            gt = ground_truth.get(key, set())
            intersection = gt & union_pred
            union_intersections.append(len(intersection))
            

        avg_intersections = sum(emb_intersections) / len(emb_intersections)
        print(f"Average intersection len for embeddings_predictions: {avg_intersections:.3f}")
        avg_intersections = sum(clu_intersections) / len(clu_intersections)
        print(f"Average intersection len for clusters_predictions: {avg_intersections:.3f}")
        avg_union_intersection = sum(union_intersections) / len(union_intersections)
        print(f"Average intersection len for union of embeddings and clusters predictions: {avg_union_intersection:.3f}")

        # non_empty_count_clusters = sum(1 for intersection_set in intersections_clusters.values() if intersection_set)
        # non_empty_count_clusters_filtered = sum(1 for intersection_set in intersections_clusters_filtered.values() if intersection_set)
        # print(f"Number of nodes with non-empty intersection using clusters: {non_empty_count_clusters} out of {len(selected_nodes)}")
        # print(f"Number of nodes with non-empty intersection using filtered clusters: {non_empty_count_clusters_filtered} out of {len(selected_nodes)}")
        for name, preds in [
            ("Embeddings", embeddings_predictions),
            ("Clusters", clusters_predictions),
            ("Union", union_predictions)
        ]:
            precision, recall, f1 = compute_set_metrics(ground_truth, preds)
            print(f"{name} - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

        for node in selected_nodes:
            print(len(ground_truth[node]), len(embeddings_predictions[node]), len(clusters_predictions[node]), len(union_predictions[node]))
if __name__ == "__main__":
    main()