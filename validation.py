import pickle
import argparse
#import clickhouse_connect
import os
from graph_utils import *
import collections

def sample_nodes_within_degree_range(graph, degree_min, degree_max, x):
    eligible_nodes = [node for node, degree in graph.degree() if degree_min <= degree < degree_max]
    # Step 2: If fewer than x nodes match, adjust x to avoid an error
    if len(eligible_nodes) < x:
        print(f"Warning: Only {len(eligible_nodes)} nodes available within the degree range.")
        x = len(eligible_nodes)
    sampled_nodes = random.sample(eligible_nodes, x)
    return sampled_nodes

def plot_degree_distribution(graph):
    degree_sequence = [d for n, d in graph.degree()]  # Get the degree of each node
    # Count how many nodes have each degree
    degree_count = collections.Counter(degree_sequence)
    degrees, counts = zip(*sorted(degree_count.items()))  # Sort by degree

    # Create the plot with customized style
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, counts, marker='o', linestyle='--', color='#FF6347', markersize=8, 
             markerfacecolor='#4682B4', linewidth=2)

    # Log-log scale for both axes
    plt.xscale('log')
    plt.yscale('log')

    # Customize x-axis and y-axis labels
    plt.xlabel('Degree', fontsize=14, labelpad=10)
    plt.ylabel('Frequency', fontsize=14, labelpad=10)
    
    # Title with consistent style
    plt.title('Frequency of Each Degree (Log-Log Scale)', fontsize=16, fontweight='bold', pad=20)

    # Grid customization (matching the second function)
    plt.grid(True)

    # Customize x and y ticks
    plt.xticks(fontsize=12, fontweight='medium')
    plt.yticks(fontsize=12, fontweight='medium')

    # Set a light gray background for consistency
    plt.gca().set_facecolor('#f5f5f5')

    # Tight layout for better spacing
    plt.tight_layout()

    # Save and show the plot
    plt.savefig('edge_degree_frequency.png')
    plt.show()

def main():

    parser = argparse.ArgumentParser(description='Insert thresholds for the graph cluster algorithm')
    parser.add_argument('--load', default= "True", help='Specify if load the graph or create a new one')
    parser.add_argument('--t_min', help='Specify the min threshold')
    parser.add_argument('--t_max', help='Specify the max threshold')
    parser.add_argument('--mode', choices=["weighted","unweighted"], help='Specify whether to use weighted or unweighted graph')
    parser.add_argument('--seed', help='Specify id of the seed node')
    args = parser.parse_args()

    if args.load == "False":
        # host = 'localhost'
        # port = 8123
        # database = 'mydb'
        # user = 'evision'
        # psw= 'Evision!'
        # table_name = 'dati_scontrini'
        
        # client = clickhouse_connect.get_client(host=host, port=port, username=user, password=psw, database=database)
        # edge_weights = calulate_edge_weights(client, table_name)
        # with open('edge_weights.pkl', 'wb') as file:
        #     pickle.dump(edge_weights, file)
    
        with open('edge_weights.pkl', 'rb') as file:
            edge_weights = pickle.load(file)
        
        #display_edge_weight_distribution(edge_weights)
        fit_powerlaw_on_edge_distribution_CDDF(edge_weights)
        exit(0)
        product_graph, product_to_index, index_to_product = create_graph(edge_weights, int(args.t_min), int(args.t_max))

        # with open("graph.pkl", "wb") as f:
        #     pickle.dump(product_graph, f)
        
        # with open('products_dict.pkl', 'wb') as f:
        #     pickle.dump({'id_to_index': product_to_index, 'index_to_id': index_to_product}, f)
    
    else:
        file_path = "graph.pkl"
        product_graph = load_graph(file_path)
    
    # with open("products_dict.pkl", 'rb') as f:
    #     data = pickle.load(f)
    #     product_to_index = data['id_to_index']
    #     index_to_product = data['index_to_id']
             
    print("Number of nodes:", product_graph.number_of_nodes())
    print("Number of edges:", product_graph.number_of_edges())
    
    percent = 30
    reduced_graph = remove_random_edges(product_graph, percent)

    print("Number of nodes:", reduced_graph.number_of_nodes())
    print("Number of edges:", reduced_graph.number_of_edges())
    validation(product_graph, reduced_graph, 1000, args.mode)

    # with open('unweighted_metrics.txt', 'w') as f:
    # #with open('weighted_metrics.txt', 'w') as f:
    #     f.write(f"t_min: {args.t_min}\n")
    #     f.write(f"t_max: {args.t_max}\n")
    #     f.write(f"std_devs: {std_devs}\n")
    #     f.write(f"metric_values: {metric_values}\n")
    #     f.write(f"epsilon_values: {epsilon_values}\n")
    #     f.write(f"accuracy: {accuracy}\n")
    #     f.write(f"accuracy_dev: {accuracy_dev}\n")
    #     f.write(f"cluster_ratio: {cluster_ratio}\n")
    #     f.write(f"cluster_ratio_dev: {cluster_ratio_dev}\n")

    #metrics_plot_with_std_devs(metric_values, epsilon_values, std_devs)
    # print(metric_values)
    # print(accuracy)
    
    # host = 'localhost'
    # port = 8123
    # database = 'mydb'
    # user = 'evision'
    # psw= 'Evision!'
    # table_name = 'dati_scontrini'
    
    # client = clickhouse_connect.get_client(host=host, port=port, username=user, password=psw, database=database)
     
    #test_different_epsilon(args, table_name, client, product_graph, index_to_product, product_to_index)

    # print("seed node:", seed)
    # #print("Identified cluster:", cluster)
    # ids = [index_to_product[index] for index in cluster]
    # print(len(ids))
    # ids = ', '.join(map(str, ids))
    # query= f"SELECT DISTINCT descr_prod FROM {table_name} WHERE cod_prod IN ({ids});"
    # result = client.query(query)
    # print("cluster products: ",result.result_rows)
    # query= f"SELECT DISTINCT descr_prod FROM {table_name} WHERE cod_prod IN ({index_to_product[seed]});"
    # result = client.query(query)
    # print("seed product:", result.result_rows)
    # client.close()

if __name__ == "__main__":
    main()