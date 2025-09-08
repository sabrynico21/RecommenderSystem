import pickle
import argparse
import clickhouse_connect
import os
import pandas as pd
from dotenv import load_dotenv
from graph_utils import *
from graph import Graph
from marketbasket import save_frequent_items
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.utils.convert import from_networkx
from LightGCN import LightGCN
from numpy.linalg import norm
import networkx as nx


def main():

    parser = argparse.ArgumentParser(description='Insert thresholds for the graph cluster algorithm')
    parser.add_argument('--load', default= "True", help='Specify if load the graph or create a new one')
    parser.add_argument('--t_min', default=0, help='Specify the min threshold')
    parser.add_argument('--t_max', default=float('inf'), help='Specify the max threshold')
    parser.add_argument('--mode', default="weighted", choices=["weighted","unweighted"], help='Specify whether to use weighted or unweighted graph')
    parser.add_argument('--seed', help='Specify id of the seed node')
    parser.add_argument('--task', default="performances", choices=["predictions", "performances","frequent_items", "prova"], help='Specify the task to perform')
    parser.add_argument('--edge_weight', default=50, type=int, help='Specify the edge weight for link prediction task')
    parser.add_argument('--random', default="False", help='Specify if use random selection of nodes/edges for the selected task')
    args = parser.parse_args()
    load_dotenv()

    client = clickhouse_connect.get_client(
            host=os.getenv('CLICKHOUSE_HOST'),
            port=int(os.getenv('CLICKHOUSE_PORT')),
            username=os.getenv('CLICKHOUSE_USER'),
            password=os.getenv('CLICKHOUSE_PASSWORD'),
            database=os.getenv('CLICKHOUSE_DATABASE')
            )

    if args.task == "prova":
        table_name = 'dati_scontrini'
        if (os.path.exists(f"../data/train_edge_weights.pkl") and os.path.exists(f"../data/test_edge_weights.pkl")):
            with open(f"../data/train_edge_weights.pkl", "rb") as f:
                train_edge_weights = pickle.load(f)
            with open(f"../data/test_edge_weights.pkl", "rb") as f:
                test_edge_weights = pickle.load(f)
        else:
            train_edge_weights, test_edge_weights = calculate_edge_weights(client, table_name, "split")
            with open(f"../data/train_edge_weights.pkl", "wb") as f:
                pickle.dump(train_edge_weights, f)
            with open(f"../data/test_edge_weights.pkl", "wb") as f:
                pickle.dump(test_edge_weights, f)
        
        train_graph = Graph()
        test_graph = Graph()
        print("arrivata qui")
        train_graph.create_graph(train_edge_weights, int(args.t_min), float(args.t_max))
        test_graph.create_graph(test_edge_weights, int(args.t_min), float(args.t_max))
        print("Train Graph - Number of nodes:", train_graph.number_of_nodes())
        print("Train Graph - Number of edges:", train_graph.number_of_edges())
        print("Test Graph - Number of nodes:", test_graph.number_of_nodes())
        print("Test Graph - Number of edges:", test_graph.number_of_edges())

        # with open(f"../data/train_graph.pkl", "wb") as f:
        #     pickle.dump(train_graph.get_graph(), f)

        # with open(f"../data/test_graph.pkl", "wb") as f:
        #     pickle.dump(test_graph.get_graph(), f)
        
        # with open(f'../data/train_products_dict.pkl', 'wb') as f:
        #     pickle.dump({'id_to_index': train_graph.get_product_to_index(), 'index_to_id': train_graph.get_index_to_product()}, f)
        
        # with open(f'../data/test_products_dict.pkl', 'wb') as f:
        #     pickle.dump({'id_to_index': test_graph.get_product_to_index(), 'index_to_id': test_graph.get_index_to_product()}, f)
        
        # file_path = f"../data/train_graph.pkl"
        # train_graph = Graph.load_graph(file_path, int(args.t_min), float(args.t_max))
        # dict_path = f"../data/train_products_dict.pkl"
        # train_graph.load_dicts(dict_path)  

        # file_path = f"../data/test_graph.pkl"
        # test_graph = Graph.load_graph(file_path, int(args.t_min), float(args.t_max))
        # dict_path = f"../data/test_products_dict.pkl"
        # test_graph.load_dicts(dict_path)  

        with open("../data/selected_nodes.txt") as f:
            selected_nodes = ast.literal_eval(f.read())
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
        #print(train_neighbors)
        #print(test_neighbors)
        print(ground_truth)
        print("ottenute etichette")

        #node_embeddings = train_lightgcn(train_graph)
        #print("Embeddings shape:", node_embeddings.shape)  # [num_nodes, 64]
        
        data = create_pyg_data_from_networkx(train_graph, weight_attr='weight')
        model = LightGCN(num_nodes=data.num_nodes, embedding_dim=64, num_layers=3)

        # Load the saved weights
        model.load_state_dict(torch.load("../LightGCN/lightgcn_model.pth"))
        model.eval()
        node_embeddings = model.get_embeddings(data.edge_index, edge_weight=data.edge_attr)
        print("Embeddings shape:", node_embeddings.shape)  # [num_nodes, 64]

        G_pruned = prune_graph(train_graph, node_embeddings, weight_threshold=15, sim_threshold=0.7)
        print("G pruned - Number of nodes:", G_pruned.number_of_nodes())
        print("G pruned - Number of edges:", G_pruned.number_of_edges())
        
        # 6. Use embeddings for recommendation (e.g., find similar products)
        def get_similar_products(product_idx, k=15):
            product_embedding = node_embeddings[product_idx]
            similarities = torch.nn.functional.cosine_similarity(product_embedding.unsqueeze(0), node_embeddings, dim=1)
            top_k = torch.topk(similarities, k=k+1)  # +1 to exclude self
            return top_k.indices[1:], top_k.values[1:]  # Return indices and scores, excluding self

        # Example: Find products similar to product 0
        selected_nodes_indices = [G_pruned.get_product_to_index(node) for node in selected_nodes]
        top_15 = defaultdict(set)
        for i,node_idx in enumerate(selected_nodes_indices):
            similar_indices, similar_scores = get_similar_products(node_idx, k=30)
            similar_indices = similar_indices.tolist()
            top_15[selected_nodes[i]] = set([G_pruned.get_index_to_product(x) for x in similar_indices])
        #similar_indices = similar_indices.tolist()
        #print(f"Products similar to product 0: {[train_graph.get_index_to_product(x) for x in similar_indices]}")
        intersections = {key: ground_truth[key] & top_15.get(key, set()) for key in set(ground_truth) | set(top_15)}
        #print("Intersections:", intersections)
        non_empty_count = sum(1 for intersection_set in intersections.values() if intersection_set)
        print(f"Number of nodes with non-empty intersection: {non_empty_count} out of {len(selected_nodes)}")

        clusters, _ = calculate_clusters(G_pruned, selected_nodes_indices, args.mode)
        top_15_clusters = defaultdict(set)
        for i, node in enumerate(selected_nodes):
            top_15_clusters[node] = set([G_pruned.get_index_to_product(x) for x in clusters[i][:min(30, len(clusters[i]))]])
        intersections_clusters = {key: ground_truth[key] & top_15_clusters.get(key, set()) for key in set(ground_truth) | set(top_15_clusters)}
        non_empty_count_clusters = sum(1 for intersection_set in intersections_clusters.values() if intersection_set)
        print(f"Number of nodes with non-empty intersection using clusters: {non_empty_count_clusters} out of {len(selected_nodes)}")
    exit(0)
    if args.load == "False":
        #table_name = 'dati_scontrini'
        # edge_weights = calulate_edge_weights(client, table_name)
        # with open('edge_weights.pkl', 'wb') as file:
        #     pickle.dump(edge_weights, file)
        
        with open('../data/edge_weights.pkl', 'rb') as file:
           edge_weights = pickle.load(file)

        #display_edge_weight_distribution(edge_weights)
        #fit_powerlaw_on_edge_distribution_CDDF(edge_weights)
    
        graph = Graph()
        graph.create_graph(edge_weights, int(args.t_min), float(args.t_max))
        #plot_degree_distribution(graph)
        #fit_powerlaw_on_degree_distribution_CDDF(graph)
        
        with open(f"../data/graph_{graph.t_min}-{graph.t_max}.pkl", "wb") as f:
            pickle.dump(graph.get_graph(), f)
        
        with open(f'../data/products_dict_{graph.t_min}-{graph.t_max}.pkl', 'wb') as f:
            pickle.dump({'id_to_index': graph.get_product_to_index(), 'index_to_id': graph.get_index_to_product()}, f)
        
    else:
        file_path = f"../data/graph_{args.t_min}-{args.t_max}.pkl"
        graph = Graph.load_graph(file_path, int(args.t_min), float(args.t_max))
        dict_path = f"../data/products_dict_{graph.t_min}-{graph.t_max}.pkl"
        graph.load_dicts(dict_path)    

    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())
    
    if args.task == "predictions":
    
        #Link Prediction task
        testing_file_path = "../Results/Link prediction/prova.txt"
        weight = args.edge_weight
        dev = 10
        num = 50
        with open('../data/edge_weights.pkl', 'rb') as file:
            edge_weights = pickle.load(file)
        random_edges = extract_random_edges(edge_weights, weight, dev, num, testing_file_path, args.random)
        
        with open(testing_file_path, 'a') as f:
            f.write(f"EDGE_WEIGHT: {weight - dev} - {weight + dev}\n")
            f.write(f"graph: {args.t_min} - {args.t_max}\n")
        df = pd.read_csv("../data/grouped_products.csv")
        link_prediction(random_edges, graph, df, testing_file_path, args.mode)

    elif args.task == "performances":
        # Performance evaluation with Reduced Graphs
        percent = 30
        reduced_graph = remove_random_edges(graph, percent)
        print("Number of nodes:", reduced_graph.number_of_nodes())
        print("Number of edges:", reduced_graph.number_of_edges())    
        validation(graph, reduced_graph, args.mode, 1000, args.random)

    elif args.task == "frequent_items":
        # Frequent Itemsets for selected nodes
        with open("../data/selected_nodes.txt") as f:
            selected_nodes = ast.literal_eval(f.read())
        
        save_frequent_items( selected_nodes)


if __name__ == "__main__":
    main()