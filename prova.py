
from graph_utils import metrics_plot_with_std_devs

epsilon_values = [0.01, 0.1, 1, 10]
unweighted_metric_values = [0.5, 0.55, 0.58, 0.6]
weighted_metric_values = [0.6, 0.63, 0.65, 0.68]
unweighted_std_devs = [0.05, 0.04, 0.03, 0.02]
weighted_std_devs = [0.06, 0.05, 0.04, 0.03]

metrics_plot_with_std_devs(unweighted_metric_values, weighted_metric_values, epsilon_values, 
                           unweighted_std_devs, weighted_std_devs)
