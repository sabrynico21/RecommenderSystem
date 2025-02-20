import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def extract_metrics_from_file(filename):
    data = {}
    current_graph = None
    current_range = None
    
    # Open the file and parse it line by line
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            
            if line.startswith("graph:"):
                # New graph configuration
                current_graph = line.split(":")[1].strip()
                data[current_graph] = {"ranges": []}  # Initialize a new graph entry
            elif line.startswith("range:"):
                # New range block
                current_range = line.split(":")[1].strip()
                data[current_graph]["ranges"].append({
                    "range": current_range,
                    "metric_values": [],
                    "std_devs": [],
                    "epsilon_values": []
                })
            elif line.startswith("accuracy:"):
                # Parse metric_values
                metric_values = eval(line.split(":")[1].strip())
                data[current_graph]["ranges"][-1]["metric_values"] = metric_values
            elif line.startswith("accuracy_dev:"):
                # Parse std_devs
                std_devs = eval(line.split(":")[1].strip())
                data[current_graph]["ranges"][-1]["std_devs"] = std_devs
            elif line.startswith("epsilon_values:"):
                # Parse epsilon_values
                epsilon_values = eval(line.split(":")[1].strip())
                data[current_graph]["ranges"][-1]["epsilon_values"] = epsilon_values
    
    return data

def extract_data_from_file(filename):
    data = {}
    current_graph = None
    current_range = None
    epsilon_values = []
    ccr = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if line.startswith("graph:"):
                if ccr != []:
                    for i, metric in enumerate(ccr):
                        data[current_graph]["ranges"][current_range][epsilon_values[i]] = metric
                    ccr = []
                # New graph configuration
                current_graph = line.split(":")[1].strip()
                data[current_graph] = {"ranges": {}}  # Initialize a new graph entry
            elif line.startswith("range:"):
                if ccr != []:
                    for i, metric in enumerate(ccr):
                        data[current_graph]["ranges"][current_range][epsilon_values[i]] = metric
                    ccr = []
                # New range block
                current_range = str(line.split(":")[1].strip())
                data[current_graph]["ranges"][current_range] = {}
            elif line.startswith("epsilon_values:"):
                epsilon_values = eval(line.split(":")[1].strip())
                for epsilon in epsilon_values:
                    data[current_graph]["ranges"][current_range][epsilon] = {}
            elif line.startswith("CLR:"):
                ccr.append(eval(line.split(":")[1].strip()))
        if ccr != []:
            for i, metric in enumerate(ccr):
                data[current_graph]["ranges"][current_range][epsilon_values[i]] = metric
            ccr = []
    return data

def generate_plots(data):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for graph, graph_data in data.items():
        plt.figure(figsize=(10, 6))

        all_epsilons = []
        for i, range_data in enumerate(graph_data['ranges']):
            epsilon = range_data['epsilon_values']
            metrics = range_data['metric_values']
            std_devs = range_data['std_devs']
            range_label = range_data['range']

            # Collect all epsilon values
            all_epsilons.extend(epsilon)

            # Plot each range with error bars
            plt.errorbar(
                epsilon, metrics, yerr=std_devs, 
                fmt='-s', capsize=5, capthick=1, elinewidth=2, 
                color=colors[i % len(colors)], ecolor=colors[i % len(colors)], 
                markerfacecolor='#191970', marker="x", markersize=8, linestyle='-', 
                linewidth=2, label=f'Range {range_label}', alpha=0.8
            )

        # Set the x-ticks on a log scale
        plt.xscale('log')
    
        # Add labels, title, and grid
        plt.title(f"Exploring Sensitivity Across Epsilon Values", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Epsilon Values (Log Scale)", fontsize=14, labelpad=10)
        plt.ylabel("Sensitivity", fontsize=14, labelpad=10)
        plt.grid(True)
        plt.yticks(fontsize=12, fontweight='medium')
        plt.gca().set_facecolor('#f5f5f5')
        plt.legend(fontsize=12, loc='best', fancybox=True, shadow=True)
        plt.tight_layout()

        for spine in plt.gca().spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)

        # Save and show the plot
        plt.savefig(f'w_sensitivity_plot_{graph}.png')
        plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Use a Seaborn style for better aesthetics
sns.set(style="whitegrid")

def box_plot(data):
    #colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    colors = ['#ff7f0e']
    for graph, graph_data in data.items():
        for i, range_data in enumerate(graph_data['ranges'].keys()):
            # Extract epsilon values and metrics for the current range
            epsilon_values = list(graph_data['ranges'][range_data].keys())
            metrics = list(graph_data['ranges'][range_data].values())
            
            # Create figure with customized dimensions
            plt.figure(figsize=(5, 5))
            
            # Generate box plot
            box = plt.boxplot(metrics, labels=epsilon_values, patch_artist=True, showmeans=True)

            # Customize the colors of each box
            #colors = sns.color_palette("pastel", len(epsilon_values))
            for patch in box['boxes']:
                patch.set_facecolor(colors[i])
            for median in box['medians']:
                median.set_color('yellow')       # Colore rosso per la linea mediana
                median.set_linewidth(2) 

            # Visualizza la media come punto blu per ogni box
            for mean in box['means']:
                mean.set_marker('x')          # Imposta il simbolo della media
                #mean.set_markerfacecolor('black')  # Colore blu per la media
                mean.set_markeredgecolor('#333333') # Bordo nero per il punto della media
                mean.set_markersize(8) 

            # Label and title each plot with the range
            plt.xlabel('Epsilon', fontsize=12)
            plt.ylabel('CLR', fontsize=12)
            plt.title(f'Box Plot of CLR by Epsilon Value', fontsize=14, weight='bold')
            
            plt.yscale('log')
            # Add grid and show the plot
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.savefig(f"CLR{graph}_{range_data}")
            plt.show()

data = extract_data_from_file("unw_testing.txt")
box_plot(data)
#generate_plots(data)

import matplotlib.pyplot as plt
import scipy.stats as stats

# data = [0.6666666666666666, 1.0, 0.6666666666666666, 1.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.75, 0.8, 0.8571428571428571, 1.0, 1.0, 0.6595744680851063, 0.7024793388429752, 1.0, 1.0, 1.0, 0.5, 0.625, 0.0, 1.0, 0.5357142857142857, 0.75, 0.6666666666666666, 1.0, 0.3333333333333333, 0.8571428571428571, 0.3333333333333333, 0.2, 0.4931506849315068, 0.8095238095238095, 0.6086956521739131, 0.0, 0.6428571428571429, 0.0, 0.0, 1.0, 0.5625, 0.6, 0.8181818181818182, 0.3918918918918919, 0.6, 0.7142857142857143, 0.9642857142857143, 0.5, 0.6153846153846154, 0.5, 0.6388888888888888, 0.6666666666666666, 0.6, 0.8888888888888888, 1.0, 0.4, 0.0, 1.0, 0.5, 0.8064516129032258, 0.3333333333333333, 0.8333333333333334, 0.5633802816901409, 0.4166666666666667, 0.3333333333333333, 0.0, 0.5, 0.6071428571428571, 0.8913043478260869, 0.4, 0.0, 1.0, 0.0, 0.4117647058823529, 0.0, 0.0, 0.125, 0.96875, 0.8666666666666667, 0.5, 0.375, 0.9090909090909091, 0.4444444444444444, 0.9, 1.0, 1.0, 1.0, 1.0, 0.7037037037037037, 1.0, 0.45454545454545453, 0.0, 0.9464285714285714, 0.5, 0.5490196078431373, 0.8, 1.0, 0.6, 0.8571428571428571, 0.7272727272727273, 0.6216216216216216, 0.5, 0.0, 0.75, 0.0, 1.0, 1.0, 0.0, 0.175, 0.6666666666666666, 0.3333333333333333, 0.6206896551724138, 1.0, 0.3728813559322034, 0.7083333333333334, 1.0, 0.0, 0.75, 0.577639751552795, 0.5, 0.5161290322580645, 1.0, 0.6896551724137931, 0.0, 0.0, 0.7692307692307693, 1.0, 0.6666666666666666, 0.5, 0.5833333333333334, 0.6, 0.6585365853658537, 0.7894736842105263, 0.0, 0.8285714285714286, 0.6153846153846154, 0.0, 0.48717948717948717, 0.5555555555555556, 0.5, 0.7931034482758621, 0.5882352941176471, 0.0, 0.6, 0.0, 0.0, 0.6666666666666666, 0.8181818181818182, 0.0, 0.6666666666666666, 1.0, 0.9217391304347826, 1.0, 0.7142857142857143, 0.0, 0.5714285714285714, 1.0, 0.75, 0.0, 0.9090909090909091, 0.7857142857142857, 0.21428571428571427, 0.0, 1.0]
# stats.probplot(data, dist="norm", plot=plt)
# plt.title("Q-Q Plot")
# plt.show()

# data = extract_data_from_file("unweighted_metrics.txt")
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
# for graph, graph_data in data.items():
#     fig, ax1 = plt.subplots(figsize=(10, 6))

#     # Create ax2 outside the loop, so it's only created once per graph
#     ax2 = ax1.twinx()

#     # We will store handles and labels for each type of plot to create legends later
#     lines_range = []
#     lines_type = []

#     for i, range_data in enumerate(graph_data['ranges']):
#         single_node = range_data['single_node']
#         epsilon = range_data['epsilon_values']
#         metrics = range_data['metric_values']
#         std_devs = range_data['std_devs']
#         range_label = range_data['range']
    
#         # Plot the metrics with error bars on ax1
#         line1 = ax1.errorbar(
#             epsilon, metrics, yerr=std_devs, 
#             fmt='-s', capsize=5, capthick=1, elinewidth=2, 
#             color=colors[i], ecolor=colors[i], markerfacecolor='#191970', marker="x",
#             markersize=8, linestyle='-', linewidth=2, alpha=0.8, label=f'Range {range_label}'
#         )

#         # Plot the single node data on ax2
#         line2, = ax2.plot(epsilon, single_node, 'r--o', color=colors[i], label=f'Single Node Clusters {range_label}')

#     # Add labels for types of lines (for the second legend)
#     lines_type.append(ax1.plot([], [], 'k-', label='Mean Cluster Length')[0])  # Solid line
#     lines_type.append(ax2.plot([], [], 'k--', label='Single Node Clusters')[0])  # Dashed line

#     ax1.set_xscale('log')
#     ax1.set_yscale('symlog')
#     ax1.set_xlabel('Epsilon values', fontsize=14, labelpad=10)
#     ax1.set_ylabel('Mean Cluster Length', fontsize=14, labelpad=10)
#     ax1.tick_params(axis='y')

#     ax2.set_ylabel('Single Node Clusters', fontsize=14, labelpad=10)
#     ax2.tick_params(axis='y')

#     plt.title(f'Cluster Mean Length and Single Node Clusters vs Epsilon', fontsize=16, fontweight='bold', pad=20)
#     plt.grid(True)
#     plt.yticks(fontsize=12, fontweight='medium')
#     plt.gca().set_facecolor('#f5f5f5')

#     # First legend for "range" based on color (custom legend with only colors)
#     legend1_labels = [f'Range {range_data["range"]}' for range_data in graph_data['ranges']]
#     legend1_handles = [Line2D([0], [0], color=colors[i], marker='o', linestyle='None', markersize=10) for i in range(len(legend1_labels))]
#     ax1.legend(legend1_handles, legend1_labels, loc='best', fontsize=12, fancybox=True)

#     # Second legend for "type" based on line style (for mean cluster length and single node clusters)
#     ax2.legend(handles=lines_type, loc='best', fontsize=12, fancybox=True)

#     plt.tight_layout()
#     #plt.savefig(f'mean_len_plot_{graph}.png')
#     plt.show()