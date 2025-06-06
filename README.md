# Random-walk-based Recommender System

This repository contains the code and experiments for the paper **"Real-Time Personalized Recommendations Using Local Random Walks on Co-Purchase Networks"**, which proposes a novel approach to recommendation using random walks on graphs.
The paper introduces a graph-based method that leverages the topology of item co-occurrence for recommendations, aiming to demonstrate how random walks can reveal item-item relationships in a grocery retail domain.

## 🧠 Overview of the Method

- We model the recommendation problem as a graph, where nodes represent products, and an edge connects two products if they appear together in at least one receipt.
- Our approach relies on a modified version of the PageRank-Nibble algorithm. Starting from a given product node, it computes an Approximate Personalized PageRank vector to sample and rank related products.

## 📁 Repository Structure

<pre>
├── data/                        # Graph data used in experiments
│   ├── graph_*.pkl              # Pickled NetworkX graphs
│   ├── edge_weights.pkl         # Pickled dictionary containing co-purchase product frequencies
│   ├── selected_nodes.txt       # Nodes selected to evaluate performance (CCR, F1-score, execution time)

├── output/frequent_items        # Frequent items obtained by executing the FP-Growth algorithm
 
├── src/                         # Source code
│   ├── validation.py            # Main script to run experiments
│                                # Args:
│                                #   --load (bool): If True, load graph from .pkl
│                                #   --mode (str): 'weighted' or 'unweighted'
│                                #   --t_min / --t_max (float): Edge weight thresholds
│                                #   --task (str): 'predictions', 'performances', 'frequent_items'
│                                #   --edge_weight (int): Weight used to sample edges (for link prediction)
│                                #   --random (bool): If True, select nodes/edges randomly to perform the selected experiment  
│   ├── display_plots.py         # Script for visualizing results

├── results/                     # Output from experiments
│   ├── Link prediction/         # Visuals and metrics for the link prediction task
│   ├── Frequent Itemset Comparison/
│                                # Comparison with frequent itemset mining methods
│   ├── w_test_epsilon_performances.pkl
│   └── unw_test_epsilon_performances.pkl
│                                # Metrics: CCR, F1-score, execution time (weighted/unweighted)

├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
</pre>

## 🧪 Running Experiments

You can run different experiments using the `validation.py` script by passing the appropriate command-line arguments. If the `--random` parameter is set to "True", the same nodes/edges used in the original paper will be selected.

### 📊 Performances

Run the algorithm and evaluate it using CCR, F1-score, and execution time:

```bash
python -m validation --load True --mode <weighted|unweighted> --t_min <int> --t_max <int> --task performances --random <True|False> 
```
### 🔗 Link prediction

Evaluate the algorithm’s performance in link prediction by removing specific co-purchased product information (edge), along with all transactions (receipts) in which both products appear, from the recommendation graph.

```bash
python -m validation --load True --mode <weighted|unweighted> --t_min <int> --t_max <int> --task predictions --edge_weight 100 --random <True|False>
```
### 🧺 Frequent Items Calculation

Compute the frequent itemsets for the selected products (listed in selected_products.txt) to compare with the results of the random-walk-based approach:

```bash
python -m validation --load True --mode <weighted|unweighted> --t_min <int> --t_max <int> --task frequent_items


