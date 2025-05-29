import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import powerlaw
from src.pagerank_nibble import page_rank_nibble
from collections import Counter, OrderedDict
from itertools import combinations
from collections import defaultdict
import collections
#random.seed(42)

def link_prediction(random_edges, graph, client, testing_file_path):
    or_same_cluster = []
    re_same_cluster = []
    total_jaccard_sim = []
    total_or_mean = []
    total_red_mean = []
    
    for i in range(len(random_edges)):
        red_graph = graph.new_graph_removing_receipts(client, random_edges[i])
        print("Number of edges:", graph.number_of_edges())
        print("Number of edges:", red_graph.number_of_edges())
        product1 = graph.get_product_to_index(random_edges[i][0])
        product2 = graph.get_product_to_index(random_edges[i][1])
        or_count, red_count, jaccard_similarity, or_mean, red_mean = compare_clusters(graph, red_graph,[product1, product2], args.mode, 0.0001)
        or_same_cluster.append(or_count)
        re_same_cluster.append(red_count)
        total_jaccard_sim.append(jaccard_similarity)
        total_or_mean.append(or_mean)
        total_red_mean.append(red_mean)
        print(or_count, red_count, jaccard_similarity)
    with open(testing_file_path, 'a') as f:
        f.write(f"or_cluster: {or_same_cluster}\n")
        f.write(f"re_cluster: {re_same_cluster}\n")
        f.write(f"jaccard_sim: {total_jaccard_sim}\n")
        f.write(f"or_edge_weights_mean: {total_or_mean}\n")
        f.write(f"red_edge_weights_mean: {total_red_mean}\n")

def plot_degree_distribution(graph):
    degree_sequence = [d for n, d in graph.degree()]
    degree_count = collections.Counter(degree_sequence)
    degrees, counts = zip(*sorted(degree_count.items())) 

    plt.figure(figsize=(14, 6))
    plt.plot(degrees, counts, marker='o', linestyle='', color='#FF6347', markersize=8, 
             markerfacecolor='#4682B4', linewidth=2)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Degree', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)  
    #plt.title('Degree Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.grid(True)
    plt.xticks(fontsize=12, fontweight='medium')
    plt.yticks(fontsize=12, fontweight='medium')
    plt.tight_layout()
    plt.savefig(f'{graph.t_min}-{graph.t_max}_edge_degree_frequency.png')
    plt.show()

def extract_random_edges(data, val, dev, num, file_path):
    filtered_keys = [k for k, v in data.items() if val - dev <= v <= val + dev]
    local_random = random.Random(42)
    if len(filtered_keys) < num:
        with open(file_path, 'a') as f:
            f.write(f"trovati solo {len(filtered_keys)} nodi \n")
    return local_random.sample(filtered_keys, min(num, len(filtered_keys)))

def query(client):
    query = f"SELECT DISTINCT(products) FROM grouped_products WHERE products NOT LIKE '% %';"
    result = client.query(query)
    print(len(result.result_rows)) 

def calulate_edge_weights(client, table_name):
    query = f"SELECT id_sc, arrayStringConcat(groupArray(cod_prod), ' ') AS products FROM {table_name} GROUP BY id_sc;"
    result = client.query(query)
    edge_weights = defaultdict(int)
    for row in result.result_rows: 
        products = list(set(row[1].split(' ')))
        #print(products)
        for i in range(len(products)):
            for j in range(i + 1, len(products)):
                edge = (products[i], products[j])
                edge_weights[edge] += 1
    return edge_weights

def display_edge_weight_distribution(edge_weights):
    weights = [weight for _, weight in edge_weights.items()]
    weight_counts = Counter(weights)
    sorted_weight_counts = OrderedDict(sorted(weight_counts.items()))

    weights = list(sorted_weight_counts.keys())
    print(weights[-1])
    counts = list(sorted_weight_counts.values())

    # Create the plot with customized style
    plt.figure(figsize=(14, 6))
    plt.plot(weights, counts, marker='o', linestyle='', color='#FF6347', markersize=8, 
             markerfacecolor='#4682B4', linewidth=2)

    # Log-log scale for both axes
    plt.xscale('log')
    plt.yscale('log')

    # Customize x-axis and y-axis labels
    plt.xlabel('Weights', fontsize=14, labelpad=10)
    plt.ylabel('Frequency', fontsize=14, labelpad=10)
    
    # Title with consistent style
    plt.title('Edge Weight Distribution', fontsize=16, fontweight='bold', pad=20)

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
    plt.savefig('edge_weights_frequency_custom.png')
    plt.show()

def print_fit_power_law_on_CCDF(values, counts, empirical_ccdf, title, name_plot):
    expanded_values = np.repeat(values, counts)
    print(f"Length of expanded values: {len(expanded_values)}")
    fit = powerlaw.Fit(expanded_values, discrete=True)
    alpha = fit.power_law.alpha
    xmin = fit.power_law.xmin

    print(f'Power law exponent: {alpha}')
    print(f'xmin: {xmin}')

    plt.figure(figsize=(14, 6))
    fit.power_law.plot_pdf(color='r', linestyle='--', label=f'Power law fit (xmin={xmin:.2f}, alpha={alpha:.2f})')
   
    #plt.step(values, empirical_ccdf, where='post', marker='o', markerfacecolor='yellow', linestyle='', color='b', label='Empirical data')
    plt.plot(values, counts, marker='o', linestyle='', color='#FF6347', markersize=8, 
             markerfacecolor='#4682B4', linewidth=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{name_plot}.png')
    plt.show()

def print_fit_power_law(values, counts, name_plot):
    expanded_values = np.repeat(values, counts)
    print(f"Length of expanded values: {len(expanded_values)}")

    # Fit the power law to the expanded values
    fit = powerlaw.Fit(expanded_values, discrete=True, xmin= 10)
    alpha = fit.power_law.alpha
    xmin = int(fit.power_law.xmin)
    #xmax = int(fit.power_law.xmax)

    print(f'Power law exponent: {alpha}')
    print(f'xmin: {xmin}')

    # Create a figure for plotting
    plt.figure(figsize=(14, 6))

    # Create the first axis (ax1) for frequency (counts)
    ax1 = plt.gca()

    ax1.plot(values, counts, marker='o', linestyle='', color='gold', markersize=8, 
             markerfacecolor='#4682B4', linewidth=2, label='Empirical Data')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Degree Values', fontsize=14)
    ax1.set_ylabel('Frequency', fontsize=14)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)

    ax2 = ax1.twinx()
    fit.power_law.plot_pdf(color='r', linestyle='--', label = fr'Power law fit ($x_{{\mathrm{{min}}}}$ = {xmin}, α = {alpha:.2f})', ax=ax2)
    ax2.set_ylabel('PDF', fontsize=14)
    ax2.tick_params(axis='y', labelsize=12)

    # Title and legends
    #plt.title(title, fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True)

    # Show the legend
    ax1.legend(loc='lower left', fontsize=14)
    ax2.legend(loc='upper right', fontsize=14)

    # Save the plot
    plt.savefig(f'{name_plot}.png')

    # Display the plot
    plt.show()

def fit_powerlaw_on_degree_distribution(G):
    degrees = [degree for _, degree in G.degree()]
    degree_counts = Counter(degrees)
    sorted_degree_counts = OrderedDict(sorted(degree_counts.items()))
    degrees = np.array(list(sorted_degree_counts.keys()))
    counts = np.array(list(sorted_degree_counts.values()))
    # Calculate empirical CCDF
    #empirical_ccdf = 1.0 - np.cumsum(counts) / np.sum(counts)
    #print_fit_power_law_on_CCDF(degrees, counts, empirical_ccdf, "Power Law Fit on Degree Distribution", "power_law_fit_degree")
    print_fit_power_law(degrees, counts, "power_law_fit_degree")

def fit_powerlaw_on_edge_distribution(edge_weights):
    weights = [weight for _, weight in edge_weights.items()]
    weight_counts = Counter(weights)
    sorted_weight_counts = OrderedDict(sorted(weight_counts.items()))
    weights = np.array(list(sorted_weight_counts.keys()))
    counts = np.array(list(sorted_weight_counts.values()))
    #empirical_ccdf = 1.0 - np.cumsum(counts) / np.sum(counts)  
    print_fit_power_law(weights, counts, "power_law_fit_edgeweight")

def remove_random_edges(graph, percentage):
    graph_copy = graph.copy()
    num_edges_to_remove = int(percentage * graph.number_of_edges() / 100)
    edges = list(graph.edges())   
    edges_to_remove = random.sample(edges, num_edges_to_remove)
    graph_copy.remove_edges_from(edges_to_remove)    
    return graph_copy

def calculate_clusters(graph, reduced_graph, selected_nodes, mode, c):
    beta = 0.85
    #c = 0.1
    #epsilon = 2e-05
    original_cluster = []
    reduced_cluster = []
    for node in selected_nodes:
        seed, cluster = page_rank_nibble(graph, beta, c, mode, node)
        print("or len: ", len(cluster))
        original_cluster.append(cluster)
        seed, cluster = page_rank_nibble(reduced_graph, beta, c, mode, node)
        print("red len: ", len(cluster))
        reduced_cluster.append(cluster)
    return original_cluster, reduced_cluster

def metric_calculation(graph, original_cluster, reduced_cluster):
    result = []
    sensitivity = []
    precision = []
    #cluster_ratio = []
    jaccard_sim = []
    for or_cluster, red_cluster in zip(original_cluster, reduced_cluster):
        den = num = 0
        for u, v in combinations(or_cluster, 2):
            if graph.has_edge(u, v):
                den+= 1
                elements_present = np.isin([u, v], red_cluster)
                if elements_present.all():
                    num+= 1
        if den != 0:
            result.append(num / den)
        #print("or", len(or_cluster))
        count = sum(1 for elem in red_cluster if elem in or_cluster)
        sensitivity.append(count / len(or_cluster))
        precision.append(count / len(red_cluster))
        #cluster_ratio.append(len(red_cluster) / len(or_cluster))
        jaccard_sim.append(compute_jaccard_sim(or_cluster, red_cluster))
        or_mean_degree = graph.mean_degree(or_cluster) 
        red_mean_degree = graph.mean_degree(red_cluster) 
    return result, sensitivity, precision, jaccard_sim, or_mean_degree, red_mean_degree

def sample_nodes_within_degree_range(graph, degree_min, degree_max, x, mode):
    eligible_nodes = [node for node, degree in graph.degree() if degree_min < degree <= degree_max]
    # Step 2: If fewer than x nodes match, adjust x to avoid an error
    if len(eligible_nodes) < x:
        with open('test_epsiloncompute.txt', 'a') as f:
            f.write(f"Warning: Only {len(eligible_nodes)} nodes available within the degree range.")
        x = len(eligible_nodes)
    local_random = random.Random(42)
    sampled_nodes = local_random.sample(eligible_nodes, x)
    return sampled_nodes

def are_present(cluster, products):
    print("cluster:", cluster)
    print("products:", products)
    if products[0] in cluster and products[1] in cluster:
        return True
    return False
    
def compute_jaccard_sim(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1 & set2  
    union = set1 | set2
    return len(intersection) / len(union) if len(union) > 0 else 0

def compute_mean_edge_weights(graph, cluster):
    edge_weights = graph.get_subgraph(cluster).size(weight='weight')
    num_edges = graph.get_subgraph(cluster).number_of_edges()
    return edge_weights / num_edges if num_edges > 0 else 0

def compare_clusters(graph, reduced_graph, products, mode, epsilon):
    or_cluster, red_cluster = calculate_clusters(graph, reduced_graph, [products[0]], mode, epsilon)
    print("lengths:", len(or_cluster[0]), len(red_cluster[0]))
    or_count = 0 
    red_count = 0
    jaccard_similarity = []
    for o_c, r_c in zip(or_cluster, red_cluster):
        if are_present(o_c, products): 
            or_count+=1
        if are_present(r_c, products): 
            red_count+=1
        jaccard_similarity.append(compute_jaccard_sim(o_c,r_c))
    or_mean = compute_mean_edge_weights(graph, or_cluster[0])
    red_mean = compute_mean_edge_weights(reduced_graph, red_cluster[0])
    return or_count, red_count, jaccard_similarity, or_mean, red_mean

def validation(graph, reduced_graph, num_nodes, mode):
    degree_min = [10]
    degree_max = [float('inf')]
    c = 0.03
    for min, max in zip(degree_min, degree_max):
        selected_nodes = sample_nodes_within_degree_range(graph, min, max, int(num_nodes /len(degree_min)), mode) 
        #selected_nodes = [1340, 3827, 620, 2454, 3668, 5271, 6142, 2320, 4231, 5396, 456, 4238] # 5890, 654, 2202, 3366, 13, 1844] #76, 416, 5253, 787, 10099, 4990, 4341, 4847, 918, 3334, 311, 86, 4483, 5500, 2666, 4978, 2503, 712, 3303, 6080, 532, 7572, 428, 1867, 2720, 6173, 662, 6217, 799, 3894, 1849, 1924, 429, 3960, 3266, 3899, 2634, 6024, 1283, 6588, 3244, 3947, 2878, 2376, 1134, 3821, 2900, 3011, 4529, 612, 4232, 922, 2324, 223, 4291, 480, 534, 2870, 401, 8158, 3175, 8269, 2086, 2174, 1528, 2757, 45, 1155, 15, 3935, 2, 3864, 3261, 965, 3512, 1618, 1946, 2772, 2137, 4110, 7583, 1171, 259, 4137, 2356, 6899, 3847, 391, 3178, 8342, 1281, 2434, 2745, 235, 6956, 4886, 3084, 27, 4412, 4101, 346, 529, 1538, 1932, 640, 2624, 4660, 4720, 3037, 910, 5505, 4408, 883, 7411, 1873, 5347, 7671, 322, 1200, 6167, 388, 4884, 8179, 719, 5049, 5019, 2421, 5371, 50, 9310, 6476, 3848, 2715, 824, 5763, 2445, 4518, 4204, 699, 6355, 1673, 2728, 5407, 4064, 178, 6157, 5178, 4277, 5501, 3826, 5242, 3618, 101, 6303, 954, 4452, 542, 473, 4339, 730, 339, 5259, 3948, 2164, 4380, 3379, 2418, 4597, 5237, 471, 132, 5252, 1070, 2183, 2315, 1204, 3055, 6659, 277, 2555, 2524, 2896, 545, 6026, 808, 7271, 4297, 1252, 3741, 1341, 1609, 675, 4971, 2644, 3143, 6664, 6215, 6995, 2495, 724, 3626, 2595, 1635, 4061, 1049, 55, 3576, 8410, 1792, 1779, 5011, 6194, 6798, 2074, 1291, 40, 2823, 5047, 2258, 2253, 921, 630, 5771, 3408, 4262, 6211, 613, 2682, 1568, 4420, 3469, 5274, 2570, 2331, 325, 6342, 8779, 4723, 2416, 4317, 91, 5250, 7641, 1744, 1007, 2260, 2742, 5886, 2955, 657, 3100, 120, 3568, 1157, 5533, 2866, 2380, 2013, 1754, 1145, 5270, 1942, 5342, 2131, 963, 4218, 3173, 8442, 5462, 1137, 3029, 526, 1509, 2278, 4283, 558, 4719, 1839, 1205, 275, 2926, 2419, 1732, 553, 4374, 373, 1058, 2550, 3045, 5155, 1084, 2838, 4041, 1970, 5857, 3691, 4145, 3139, 987, 4346, 2129, 6358, 552, 892, 5184, 5577, 3222, 8995, 1285, 2272, 1244, 1748, 4350, 278, 3810, 2776, 7121, 180, 3385, 3686, 2603, 3229, 402, 701, 1811, 3991, 3955, 7017, 4140, 1588, 3381, 973, 4925, 1056, 576, 635, 5136, 336, 10199, 4928, 3293, 1230, 1128, 4363, 2279, 779, 5475, 3817, 2738, 1302, 370, 1805, 2427, 1747, 4344, 4032, 6387, 637, 34, 5223, 25, 4397, 1558, 7029, 930, 4911, 3112, 8913, 3527, 4783, 969, 2523, 2788, 369, 735, 6102, 2453, 8763, 1504, 1480, 4445, 8922, 4533, 3054, 439, 2672, 4047, 3707, 1794, 1491, 3953, 579, 186, 196, 3340, 8515, 4825, 2581, 393, 152, 3034, 4185, 7507, 3571, 3342, 4052, 7192, 898, 2460, 2408, 1156, 3770, 470, 729, 1712, 358, 3721, 2928, 1865, 6766, 3505, 2684, 3170, 3118, 1069, 876, 1260, 1008, 733, 4549, 2436, 4273, 1481, 2205, 3130, 6618, 2816, 2636, 5024, 4282, 6423, 7894, 3850, 4123, 6218, 7154, 81, 3361, 1939, 3305, 2540, 3226, 1466, 6947, 52, 5353, 1532, 5474, 1212, 4399, 5122, 5834, 618, 2244, 2598, 2066, 1220, 1960, 468, 209, 6496, 1428, 2740, 2177, 5696, 626, 796, 4501, 3339, 2305, 5992, 3586, 94, 2569, 2884, 407, 1755, 2059, 2157, 1886, 2100, 5231, 1290, 4927, 1487, 313, 3926, 6319, 6802, 1417, 706, 5346, 5907, 4392, 4754, 844, 4096, 2987, 4780, 1074, 1603, 5068, 2011, 498, 2849, 1103, 2065, 291, 2353, 2744, 379, 820, 1812, 4303, 2960, 4885, 2743, 2270, 2612, 4737, 6621, 652, 2139, 780, 1063, 341, 1440, 633, 3456, 5601, 5687, 1026, 300, 115, 2512, 4413, 908, 3923, 8152, 461, 382, 1122, 2045, 535, 1088, 2590, 2739, 1099, 2942, 6474, 2538, 6645, 8008, 5613, 668, 4633, 5092, 656, 835, 6434, 795, 1182, 33, 7244, 1254, 603, 1289, 2148, 1771, 410, 7202, 8321, 5433, 7279, 1607, 331, 3089, 454, 2321, 1282, 6593, 6806, 2180, 4734, 3987, 3249, 6461, 4444, 1571, 5140, 7159, 3041, 953, 7191, 6656, 869, 1030, 4099, 3438, 1385, 4571, 3560, 4757, 6524, 1471, 2004, 8459, 2462, 3368, 3102, 2439, 3704, 3750, 2289, 2461, 988, 1853, 5560, 6021, 3404, 3959, 2399, 1086, 3338, 924, 985, 1973, 85, 4481, 4354, 3621, 1798, 2657, 1791, 776, 2415, 119, 1017, 1233, 8450, 1317, 6794, 919, 3314, 2478, 3310, 4418, 5133, 7817, 1036, 4609, 1724, 1898, 6293, 2606, 1231, 5161, 4468, 4331, 700, 1945, 3915, 490, 7491, 2271, 7674, 4059, 4523, 2648, 469, 3647, 3969, 5149, 1393, 3608, 2407, 3632, 4107, 5906, 5438, 1108, 3271, 3048, 95, 5749, 436, 710, 324, 1731, 1629, 1711, 3500, 364, 2111, 3548, 280, 6624, 1991, 2451, 4097, 3121, 660, 6420, 1038, 696, 1304, 2220, 2554, 3198, 4433, 2829, 137, 3452, 2311, 125, 328, 3417, 1262, 1987, 7230, 6344, 4033, 4829, 3742, 6419, 3433, 7407, 3957, 3638, 3702, 5838, 894, 6132, 2073, 2697, 663, 183, 1408, 3572, 6, 274, 2216, 5923, 1868, 2429, 3819, 7979, 1420, 4034, 4169, 3421, 556, 771, 998, 2200, 5982, 1598, 2334, 1596, 3207, 2843, 4640, 2382, 1649, 3620, 2237, 2014, 906, 158, 1523, 1546, 4080, 2669, 6513, 5632, 3645, 113, 3123, 1700, 5466, 288, 7108, 3485, 1248, 836, 934, 1884, 44, 4856, 3543, 2895, 4535, 7335, 218, 8168, 3360, 7701, 2358, 3155, 7025, 1413, 815, 253, 4312, 1320, 920, 2117, 2591, 527, 798, 3904, 2096, 5446, 49, 4790, 6020, 5653, 2061, 93, 2054, 4846, 8, 3117, 4442, 1628, 1075, 2409, 7173, 1018, 2905, 5105, 8988, 4416, 3376, 3598, 2560, 129, 968, 5586, 2063, 1249, 2812, 4594, 3973, 238, 1329, 736, 1982, 169, 143, 2755, 4503, 3252, 4263, 7398, 2160, 6160, 2513, 904, 2124, 2665, 3878, 523, 5706, 2162, 7617, 5750, 1424, 2909, 1077, 6620, 3976, 8651, 1684, 2268, 4441, 2578, 3237, 1034, 7089, 4733, 4883, 7488, 1136, 4621, 3038, 581, 8483, 2034, 3896, 5908, 3931, 4835, 2084, 87, 6527, 3236, 1188, 6966, 246, 5610, 1362, 1846, 1020, 2910, 1666, 5194, 5858, 5772, 621, 2229, 1723, 1482, 2721, 4540, 6245, 1728, 2916, 1441, 2882, 1185, 774, 4351, 1520, 5778, 7061, 5986, 2367, 6528, 1602, 2631, 3910, 878, 2360, 8683, 171, 1908, 4577, 1409, 4683, 333, 1835, 1014, 1277, 6149, 1713, 5937, 501, 2805, 7809, 3119, 315, 4728, 692, 2104, 1158, 1674, 5009, 4409, 2234, 4062, 3515, 5920, 3932, 3138, 3696, 5564, 3032, 241, 5406, 3699, 3193]
        single_node_cluster = []
        original_cluster, reduced_cluster = calculate_clusters(graph, reduced_graph, selected_nodes, mode, c) 
        print("len or cl:", len(original_cluster))
        print("len red cl:", len(reduced_cluster))
        result, sensitivity, precision, jaccard_sim, or_mean_degree, red_mean_degree = metric_calculation(graph, original_cluster, reduced_cluster)
        single_node_cluster = len(original_cluster) - len(result)
        len_cluster = [len(cluster) for cluster in original_cluster]
        selected_nodes = [graph.index_to_product[node] for node in selected_nodes]
        with open('test_epsiloncompute.txt', 'a') as f:
            f.write(f"constant: {c}\n")
            #f.write(f"selected_nodes: {selected_nodes}\n")
            f.write(f"CCR: {result}\n")
            f.write(f"sensitivity: {sensitivity}\n")
            f.write(f"precision: {precision}\n")
            f.write(f"lenghts: {len_cluster}\n")
            f.write(f"single_node_cluster: {single_node_cluster}\n")
            f.write(f"jaccard_similarity: {jaccard_sim}\n")
            f.write(f"or_mean_degree: {or_mean_degree}\n")
            f.write(f"red_mean_degree: {red_mean_degree}\n")
    return