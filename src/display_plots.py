from plots_utils import *

# CCR Boxplots
data = extract_metric("../Results/w_test_epsilon_performances.txt", "CCR")
unw_data = extract_metric("../Results/unw_test_epsilon_performances.txt", "CCR")
print_dual_box_plot(data.keys(), data.values(), unw_data.values(),'ccr_plots')

#F1 score Boxplots
data = extract_metrics("../Results/w_test_epsilon_performances.txt",'sensitivity', 'precision', True)
unw_data = extract_metrics("../Results/unw_test_epsilon_performances.txt", 'sensitivity', 'precision', True)
plot_f1_boxplots(data, unw_data, "f1_scores")

#Link prediction results
data = extract_data_from_file("../Results/Link prediction/w_link_prediction.txt")
unw_data = extract_data_from_file("../Results/Link prediction/unw_link_prediction.txt")
plot_link_prediction_results(data, "clusters_means")
plot_link_prediction_results(unw_data, "unw_clusters_means")


data = extract_metrics("../Results/w_test_epsilon_performances.txt", "or_clusters", "red_clusters")
unw_data = extract_metrics("../Results/unw_test_epsilon_performances.txt", "or_clusters", "red_clusters")

#Overlap Ratio Top-15 recommendations between reduced and original clusters
plot_topN(data, unw_data)

#Overlap Ratio Top-15 recommendations between weighted and unweighted clusters 
compare_originals(data, unw_data)

#Execution time comparison between weighted and unweighted graph versions
compare_execution_times(data, unw_data)

#Frequent Itemset Comparison Results
files = ['../Results/Frequent Itemset Comparison/w_clusters_40-510.pkl', '../Results/Frequent Itemset Comparison/w_clusters_15-inf.pkl']
unw_files = ['../Results/Frequent Itemset Comparison/unw_clusters_40-510.pkl', '../Results/Frequent Itemset Comparison/unw_clusters_15-inf.pkl']
final_results, unw_final_results = extract_frequent_item_recall(files, unw_files)
min_values, unw_min_values = compute_min_weights_by_graph()

display_freq_item_recall(final_results, "w_freq_items_comp", min_values)
display_freq_item_recall(unw_final_results, "unw_freq_items_comp", unw_min_values)

