import matplotlib.pyplot as plt
import itertools
import pandas as pd
import seaborn as sns
import numpy as np
from collections import Counter
from graph import Graph
import pickle
sns.set(style="whitegrid")

def extract_metric(filename, name_metric):
    data = {}
    current_graph = None
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip() 
            if line.startswith("graph:"):
                current_graph = line.split(":")[1].strip()
                data[current_graph] = []  
            elif line.startswith(f"{name_metric}:"):
                metric = eval(line.split(":")[1].strip())
                data[current_graph] =  metric
    return data

def extract_metrics(filename, metric1, metric2, f1=None):
    data = {}
    current_graph = None
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip() 
            if line.startswith("graph:"):
                current_graph = line.split(":")[1].strip()
                data[current_graph] = {}  
            elif line.startswith(f"{metric1}:"):
                metric = eval(line.split(":")[1].strip())
                data[current_graph][metric1] =  metric
            elif line.startswith(f"{metric2}:"):
                metric = eval(line.split(":")[1].strip())
                data[current_graph][metric2] = metric 
            elif line.startswith("times:"):
                metric = eval(line.split(":")[1].strip())
                data[current_graph]["time"] = metric   
                if f1 is None:
                    continue
                f1_scores = []
                for p, s in zip(data[current_graph][metric1], data[current_graph][metric2]):
                    if p + s == 0:
                        f1 = 0
                    else:
                        f1 = 2 * p * s / (p + s)
                    f1_scores.append(f1)
                data[current_graph]["f1_scores"] = f1_scores
    return data

def extract_data_from_file(filename):
    data = {}
    current_graph = None
    current_edge_weight = None
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip() 
            if line.startswith("EDGE_WEIGHT:"):
                current_edge_weight = line.split(":")[1].strip()
                data[current_edge_weight] = {"graphs": {}}  
            elif line.startswith("graph:"):
                current_graph = str(line.split(":")[1].strip())
                data[current_edge_weight]["graphs"][current_graph] = {}
            elif line.startswith("jaccard_sim:"):
                metric = eval(line.split(":")[1].strip())
                data[current_edge_weight]["graphs"][current_graph]["jaccard_sim"] =  list(itertools.chain.from_iterable(metric))
            elif line.startswith("or_cluster:"):
                metric = eval(line.split(":")[1].strip())
                data[current_edge_weight]["graphs"][current_graph]["or_cluster"] =  metric
            elif line.startswith("re_cluster:"):
                metric = eval(line.split(":")[1].strip())
                data[current_edge_weight]["graphs"][current_graph]["re_cluster"] = metric
            elif line.startswith("or_edge_weights_mean:"):
                metric = eval(line.split(":")[1].strip())
                data[current_edge_weight]["graphs"][current_graph]["or_edge_weights_mean"] = metric
            elif line.startswith("red_edge_weights_mean:"):
                metric = eval(line.split(":")[1].strip())
                data[current_edge_weight]["graphs"][current_graph]["re_edge_weights_mean"] = metric
            
    return data

def print_dual_box_plot(labels, metrics, unw_metrics, name):
    plt.figure(figsize=(7, 5))

    n = len(labels)
    positions1 = np.arange(n) - 0.2  
    positions2 = np.arange(n) + 0.2  

    box1 = plt.boxplot(metrics, positions=positions1, widths=0.3, patch_artist=True, showmeans=True)
    box2 = plt.boxplot(unw_metrics, positions=positions2, widths=0.3, patch_artist=True, showmeans=True)

    # Set colors
    color1 = '#1f77b4'  # blue
    color2 = 'mediumvioletred'  

    for patch in box1['boxes']:
        patch.set_facecolor(color1)
    for patch in box2['boxes']:
        patch.set_facecolor(color2)

    for mean in box1['means']:
        mean.set_marker('x')
        mean.set_markeredgecolor('#333333')

    for mean in box2['means']:
        mean.set_marker('x')
        mean.set_markeredgecolor('#333333')

    plt.xticks(np.arange(n), labels)
    plt.ylabel('CCR', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    #plt.grid(True)
    plt.tick_params(axis='x', pad=0)
    plt.tick_params(axis='y', pad=0)

    plt.legend([box1["boxes"][0], box2["boxes"][0]], ['Weighted graph', 'Unweighted graph'], loc='best')
    #plt.title('Comparison of Metrics vs Unweighted Metrics')
    #plt.savefig(f"{name}.png")
    plt.show()

def plot_f1_boxplots(data, unw_data, name):
    labels = []
    f1_data = []
    f1_unw_data = []
    for graph_name, metrics in data.items():
        f1_scores = metrics.get("f1_scores", [])
        if f1_scores:
            labels.append(graph_name)
            f1_data.append(f1_scores)
    for graph_name, metrics in unw_data.items():
        f1_scores = metrics.get("f1_scores", [])
        if f1_scores:
            f1_unw_data.append(f1_scores)   
    plt.figure(figsize=(7, 5))
    n = len(labels)
    positions1 = np.arange(n) - 0.2  
    positions2 = np.arange(n) + 0.2  

    box1 = plt.boxplot(f1_data, positions=positions1, widths=0.3, patch_artist=True, showmeans=True)
    box2 = plt.boxplot(f1_unw_data, positions=positions2, widths=0.3, patch_artist=True, showmeans=True)

    # Set colors
    color1 = '#1f77b4'  
    color2 = 'mediumvioletred'  

    for patch in box1['boxes']:
        patch.set_facecolor(color1)
    for patch in box2['boxes']:
        patch.set_facecolor(color2)

    for mean in box1['means']:
        mean.set_marker('x')
        mean.set_markeredgecolor('#333333')

    for mean in box2['means']:
        mean.set_marker('x')
        mean.set_markeredgecolor('#333333')

    plt.xticks(np.arange(n), labels)
    plt.ylabel('F1 Score', fontsize=14, labelpad=None)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    #plt.grid(True)
    plt.tick_params(axis='x', pad=0)
    plt.tick_params(axis='y', pad=0)

    plt.legend([box1["boxes"][0], box2["boxes"][0]], ['Weighted graph', 'Unweighted graph'], loc='best')
    #plt.title('Comparison of Metrics vs Unweighted Metrics')
    #plt.savefig(f"{name}.png")
    plt.show()

def plot_link_prediction_results(data, name):
    cmap = {0: 'red', 1: 'green'}

    for edge_weights, graph_data in data.items():
        means = []    # Edge weight means
        colors = []   # Cluster values (0 or 1)
        graphs = list(graph_data['graphs'].keys())

        # Collect data
        for graph in graphs:
            colors.append(graph_data['graphs'][graph]["or_cluster"])
            colors.append(graph_data['graphs'][graph]["re_cluster"])
            means.append(graph_data['graphs'][graph]["or_edge_weights_mean"])
            means.append(graph_data['graphs'][graph]["re_edge_weights_mean"])

        # Plot setup
        plt.figure(figsize=(8, 3))

        ytick_labels = []
        ytick_positions = []

        for line_idx in range(6): 
            config_idx = line_idx // 2
            offset = -0.2 if line_idx % 2 == 0 else 0.2
            y_pos = config_idx * 2 + offset
            values = means[line_idx]
            clusters = colors[line_idx]
            # Plot each point
            for i, (val, cl) in enumerate(zip(values, clusters)):
                plt.scatter(i, y_pos,
                            s=val, 
                            color=cmap[cl],
                            alpha=1,
                            edgecolor='k',
                            linewidth=0.3,
                            zorder=1000 - val)

            # Label every pair 
            if line_idx % 2 == 1:
                num = colors[line_idx].count(1)
                den = colors[line_idx - 1].count(1)
                label = f"{graphs[config_idx]} \n({num}/{den})"
                ytick_labels.append(label)
                ytick_positions.append(config_idx * 2)

        plt.yticks(ytick_positions, ytick_labels, fontsize=11)
        plt.grid(axis='x', linestyle=':', alpha=0.4)
        plt.tight_layout()
        #plt.savefig(f"{name}_{edge_weights}.png")
        plt.show()
    
def extract_ratio(value):
    return (2*value) / (1+value)

def get_counts(values):
    count = Counter(values)
    y_vals = list(count.keys())
    counts = list(count.values())
    return y_vals, counts

def plot_topN(data, unw_data):
    top = {}
    unw_top = {}
    for graph_name, metrics in data.items():
        original = metrics["or_clusters"]
        reduced = metrics["red_clusters"]
        for i in [15]:
            top[graph_name] = []
            for c in range(len(original)):
                set1 = set(original[c][:i])
                set2 = set(reduced[c][:i])
                intersection = set1 & set2  
                top[graph_name].append(len(intersection) / len(set1))
            
        unw_original = unw_data[graph_name]["or_clusters"]
        unw_reduced = unw_data[graph_name]["red_clusters"]
        for i in [15]:
            unw_top[graph_name] = []
            for c in range(len(unw_original)):
                set1 = set(unw_original[c][:i])
                set2 = set(unw_reduced[c][:i])
                intersection = set1 & set2  
                unw_top[graph_name].append(len(intersection) / len(set1))
                
    graph_name = list(data.keys())
    # Flatten into long-form dataframe
    all_values = (
        top[graph_name[0]] + top[graph_name[1]] + top[graph_name[2]] + top[graph_name[3]] +
        unw_top[graph_name[0]] + unw_top[graph_name[1]] + unw_top[graph_name[2]] + unw_top[graph_name[3]]
    )

    types = (
        ["weighted"] * len(top[graph_name[0]]) +
        ["weighted"] * len(top[graph_name[1]]) +
        ["weighted"] * len(top[graph_name[2]]) +
        ["weighted"] * len(top[graph_name[3]]) +
        ["unweighted"] * len(unw_top[graph_name[0]]) +
        ["unweighted"] * len(unw_top[graph_name[1]]) +
        ["unweighted"] * len(unw_top[graph_name[2]]) +
        ["unweighted"] * len(unw_top[graph_name[3]])
    )
    
    configurations = (
        [graph_name[0]] * len(top[graph_name[0]]) +
        [graph_name[1]] * len(top[graph_name[1]]) +
        [graph_name[2]] * len(top[graph_name[2]]) +
        [graph_name[3]] * len(top[graph_name[3]]) +
        [graph_name[0]] * len(unw_top[graph_name[0]]) +
        [graph_name[1]] * len(unw_top[graph_name[1]]) +
        [graph_name[2]] * len(unw_top[graph_name[2]]) +
        [graph_name[3]] * len(unw_top[graph_name[3]])
    )

    df = pd.DataFrame({
        'values': all_values,
        'type': types,
        'conf': configurations
    })

    # Plot violin
    plt.figure(figsize=(7, 5))
    sns.violinplot(data=df, x="conf", y="values", hue="type", split=True, inner="box", palette="Set2", cut=0, gap=.1)
    plt.xlabel("")
    plt.ylabel("Overlap Ratio (Top-15)", fontsize=14)
    plt.legend(title=None) 
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    #plt.savefig(f"overlap_ratio_top15.png")
    plt.show()
        
def compare_originals(data, unw_data):
    top_15 = []
    time_means = []
    unw_time_means = []
    for graph_name, metrics in data.items():
        original = metrics["or_clusters"]
        unw_original = unw_data[graph_name]["or_clusters"]
        for c in range(len(original)):
            set1 = set(original[c])
            set2 = set(unw_original[c])
            intersection = set1 & set2  
            top_15.append(len(intersection) / len(set1))
        time_means.append(np.mean(metrics["time"]))  
        unw_time_means.append(np.mean(unw_data[graph_name]["time"]))
            
    graph_names = list(data.keys())
    types = (
            [graph_names[0]] * len(original) +
            [graph_names[1]] * len(original) +
            [graph_names[2]] * len(original) +
            [graph_names[3]] * len(original)
        )
    
    df = pd.DataFrame({
            'values': top_15,
            'type': types
        })
    
    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Violin plot for overlap ratio
    sns.violinplot(data=df, x="type", y="values", inner="box", palette='pastel', ax=ax1)
    ax1.set_ylabel("Overlap Ratio", fontsize=14)
    ax1.set_ylim(0, 1.05)

    ax1.set_xlabel("")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    #plt.savefig(f"compare_originals.png")
    plt.show()
    return

def compare_execution_times(data, unw_data):
    time_means = []
    unw_time_means = []
    for graph_name, metrics in data.items():
        time_means.append(np.mean(metrics["time"]))  
        unw_time_means.append(np.mean(unw_data[graph_name]["time"]))
            
    graph_names = list(data.keys())
        
    plt.figure(figsize=(7, 5))
    
    plt.ylabel("Time (s)")

    plt.plot(time_means, color='red', marker='o', label='Weighted', linestyle='--')
    plt.plot(unw_time_means, color='blue', marker='s', label='Unweighted', linestyle='--')
    plt.xticks(ticks=range(len(graph_names)), labels=graph_names)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='best', fancybox=True, shadow=True)
    #plt.savefig(f"time_comparison.png")
    plt.show()
    return
    
    
def extract_frequent_item_recall(files, unw_files):
    with open('../Results/Frequent Itemset Comparison/frequent_items.pkl', 'rb') as f:
        frequent_items = pickle.load(f)
    final_results = {}
    unw_final_results = {}

    for i in range(len(files)):
        with open(files[i], 'rb') as f:
            data = pickle.load(f)

        with open(unw_files[i], 'rb') as f:
            unw_data = pickle.load(f)
        graph_conf = files[i].split("_")[2].split(".")[0]
        #print("graph_config", graph_conf)
        final_results[graph_conf] = { }
        unw_final_results[graph_conf] = { }
        for t in [5,10,15,20,25,30]:
            results = []
            unw_results = []
            for cluster, unw_cluster, freq_items in zip(data, unw_data, frequent_items):
                cluster = cluster[:min(t,len(cluster))]
                unw_cluster = unw_cluster[:min(t,len(unw_cluster))]
                common_elements = set(cluster) & set(freq_items)
                unw_common_elements = set(unw_cluster) & set(freq_items) 
                results.append(len(common_elements)/len(freq_items))  
                unw_results.append(len(unw_common_elements)/len(freq_items))
            final_results[graph_conf][t] = results
            unw_final_results[graph_conf][t] = unw_results
    return final_results, unw_final_results

def display_freq_item_recall(results, name, min_edges):
    data = []
    t_labels = []
    positions = []
    colors = []

    config_colors = {
        "40-510": "#1f77b4",   # blue
        "15-inf": "#ff7f0e",   # orange
    }
    t_values = [5, 10, 15, 20, 25, 30]
    box_width = 0.6
    gap = 1.5  # slightly bigger gap

    pos = 0
    config_names = list(results.keys())
    config_pos = {}

    for graph_conf in config_names:
        config_pos[graph_conf] = []
        for t in t_values:
            data.append(results[graph_conf][t])
            t_labels.append(f"top {t}")
            positions.append(pos)
            colors.append(config_colors[graph_conf])
            config_pos[graph_conf].append(pos)
            pos += 1
        pos += gap

    fig, ax = plt.subplots(figsize=(14, 6))
    box = ax.boxplot(data, positions=positions, widths=box_width, patch_artist=True, showmeans=True)

    # Style boxplot
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    for median in box['medians']:
        median.set_color('yellow')
        median.set_linewidth(2)
    for mean in box['means']:
        mean.set_marker('x')
        mean.set_markeredgecolor('#333333')
        mean.set_markersize(8)

    ax.set_xticks(positions)
    ax.set_xticklabels(t_labels, rotation=45, fontsize=11)
    ax.set_ylabel("Frequent Item Recall", fontsize=12)

    # Legend for box colors
    handles = [plt.Line2D([0], [0], color=color, lw=10) for color in config_colors.values()]
    ax.legend(handles, config_colors.keys(), title="Graph configurations", loc='upper center')

    # Secondary y-axis
    ax2 = ax.twinx()
    all_min_edge_vals = []
    config_styles = {
        "40-510": {"color": "#d62728", "marker": "o"},
        "15-inf": {"color": "#d62728", "marker": "s"},  # changed color and marker
    }
    for i, graph_conf in enumerate(config_names):
        min_edge_vals = [min_edges[i][t] for t in t_values]
        all_min_edge_vals.extend(min_edge_vals)
        ax2.plot(
            config_pos[graph_conf],
            min_edge_vals,
            marker=config_styles[graph_conf]["marker"],
            color=config_styles[graph_conf]["color"],
            linestyle='--',
            linewidth=2,
            markersize=6
        )

    ax2.set_ylabel("Minimum Edge Weight", fontsize=12)
    ax2.set_ylim(0, 50)
    ax2.grid(False)

    # Remove second legend (already shown in box color)
    # Improve layout
    #plt.title(name, fontsize=14)
    plt.savefig(f"../Results/Frequent Itemset Comparison/{name}.png")
    plt.tight_layout()
    plt.show()

# Compute the minimum edge weights for clusters obtained with specified thresholds.
def compute_min_edge_weights(t_min, t_max, weighted="True"):
    weight = "w" if weighted == "True" else "unw"
    with open(f'../Results/Frequent Itemset Comparison/{weight}_clusters_{t_min}-{t_max}.pkl', 'rb') as f:
        clusters = pickle.load(f)
    file_path = f"../data/graph_{t_min}-{t_max}.pkl"
    graph = Graph.load_graph(file_path, int(t_min), float(t_max) if t_max != "inf" else float('inf'))
    dict_path = f"../data/products_dict_{t_min}-{t_max}.pkl"
    graph.load_dicts(dict_path)
    min_values = {}
    for top_N in [5,10,15,20,25,30]:
        total_min_weight = []
        for cluster in clusters:
            cluster = cluster[:top_N]
            products_index = [graph.product_to_index[product] for product in cluster if product in graph.product_to_index]
            sub_g = graph.subgraph(products_index)
            min_weight = min(
                (data.get('weight', 1) for u, v, data in sub_g.edges(data=True)),
                default=None
            )
            if min_weight is None:
                continue
            total_min_weight.append(min_weight)
        min_values[top_N] = int(np.mean(total_min_weight))
  
    return min_values

def compute_min_weights_by_graph():
    min_ts = [40, 15]
    max_ts = [510, "inf"]
    min_values = []
    unw_min_values = []
    for i in range(len(min_ts)):
        print(f"t_min: {min_ts[i]}, t_max: {max_ts[i]}")
        min_values.append(compute_min_edge_weights(min_ts[i], max_ts[i], "True"))
        unw_min_values.append(compute_min_edge_weights(min_ts[i], max_ts[i], "False"))

    return min_values, unw_min_values

def plot_non_adjacent_proportion(portion_non_connected, model):
        portion_non_connected = np.array(portion_non_connected)

        # Safety check: make sure it can be split into 4 groups
        if len(portion_non_connected) % 4 != 0:
            raise ValueError(
                f"Expected portion_non_connected to be divisible by 4, got {len(portion_non_connected)}"
            )

        groups = np.split(portion_non_connected, 4)  # 4 equal groups
        group_labels = ['Degree 1-10', 'Degree 10-100', 'Degree 100-1000', 'Degree 1000+']

        colors = ['#fc8d62','#8da0cb','#e78ac3', '#66c2a5']  # Distinct colors for each group

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
        plt.savefig(f"non_connected_proportion_{model}.png")
        plt.show()