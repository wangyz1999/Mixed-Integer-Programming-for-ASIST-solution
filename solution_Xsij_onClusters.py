from __future__ import print_function
from ortools.linear_solver import pywraplp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

import json
from random import shuffle

from environment import MapParser
import visualizer

import time



def get_distance_matrix(graph, node_list):
    distance_matrix = np.zeros((len(node_list), len(node_list)))
    for n1 in range(len(node_list)-1):
        for n2 in range(n1, len(node_list)):
            length = nx.dijkstra_path_length(graph, node_list[n1], node_list[n2])
            distance_matrix[n1][n2] = length
    distance_matrix = distance_matrix + distance_matrix.transpose()
    return distance_matrix.tolist()


def get_distance_matrix_for_clusters(graph, clusters):
    distance_matrix = np.zeros((len(clusters)+1, len(clusters)+1))
    for n1 in range(len(clusters)-1):
        for n2 in range(n1, len(clusters)):
            length = nx.dijkstra_path_length(graph, graph[clusters[n1][0]], graph[clusters[n2][0]])
            distance_matrix[n1][n2] = length
    distance_matrix = distance_matrix + distance_matrix.transpose()
    return distance_matrix.tolist()

with open('data\\json\\Falcon_v1.0_Medium_sm_clean.json') as f:
    graph_json_data = json.load(f)

graph = MapParser.parse_json_map_data_new_format(graph_json_data)
victim_lists = graph.victim_list.copy()
start = "ew"
node_list = [graph[start]] + victim_lists

distance_matrix = get_distance_matrix(graph, node_list)
model = AgglomerativeClustering(distance_threshold=60, n_clusters=None, linkage="complete", affinity='precomputed')
model = model.fit(distance_matrix)

print(model.labels_)

clusters = [[] for i in range(max(model.labels_)+1)]
for idx, n in enumerate(node_list):
    clusters[model.labels_[idx]].append(n.id)

print(clusters)

clusters_sep_color = []
for l in clusters:
    gvl = []
    yvl = []
    for v in l:
        if 'vg' in v:
            gvl.append(v)
        if 'vy' in v:
            yvl.append(v)
    if len(yvl) > 0:
        clusters_sep_color.append(yvl)
    if len(gvl) > 0:
        clusters_sep_color.append(gvl)

clusters_sep_color = [[start]] + clusters_sep_color

print(clusters_sep_color)

s_size = len(clusters_sep_color) + 1
n_size = len(clusters_sep_color) + 1


solver = pywraplp.Solver.CreateSolver('CBC')
x = {}
for s in range(s_size):
    for i in range(n_size):
        for j in range(n_size):
            x[f"{s}_{i}_{j}"] = solver.BoolVar(f"x[{s}_{i}_{j}]")


for s in range(s_size):
    constraint_expr = [x[f"{s}_{i}_{j}"] for i in range(n_size) for j in range(n_size)]
    solver.Add(sum(constraint_expr) == 1)
#
for i in range(n_size):
    constraint_expr = []
    for s in range(s_size):
        for j in range(n_size):
            if i != j:
                constraint_expr.append(x[f"{s}_{i}_{j}"])
    solver.Add(sum(constraint_expr) == 1)
#
for j in range(n_size):
    constraint_expr = []
    for s in range(s_size):
        for i in range(n_size):
            if i != j:
                constraint_expr.append(x[f"{s}_{i}_{j}"])
    solver.Add(sum(constraint_expr) == 1)

for s in range(s_size-1):
    for j in range(n_size):
        prev_node = []
        next_node = []
        for i in range(n_size):
            prev_node.append(x[f"{s}_{i}_{j}"])
        for k in range(n_size):
            next_node.append(x[f"{s+1}_{j}_{k}"])
        solver.Add(sum(prev_node) - sum(next_node) == 0)

solver.Add(x[f"0_{len(clusters_sep_color)}_0"] == 1)

distance_matrix = get_distance_matrix_for_clusters(graph, clusters_sep_color)

obj_expr = [distance_matrix[i][j] * x[f"{s}_{i}_{j}"] for s in range(s_size) for i in range(n_size) for j in range(n_size)]
solver.Minimize(solver.Sum(obj_expr))

solver.SetTimeLimit(100000)
status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print('Objective value =', solver.Objective().Value())
    print('Problem solved in %f milliseconds' % solver.wall_time())
    print('Problem solved in %d iterations' % solver.iterations())
    print('Problem solved in %d branch-and-bound nodes' % solver.nodes())

    curr = 0
    full_path = []
    victim_path = [start]
    for s in range(s_size-1):
        for j in range(n_size):
            if int(x[f"{s}_{curr}_{j}"].solution_value()) == 1:
                curr = j
                victim_path += clusters_sep_color[j]
    print(victim_path)
    for i in range(len(victim_path)-1):
        full_path += list(map(lambda x:x.id, nx.dijkstra_path(graph, graph[victim_path[i]], graph[victim_path[i+1]])))
    print(full_path)
    visualizer.animate_graph_training_json(full_path, graph_json_data, with_save="MIP_clustering")

else:
    print('The problem does not have an optimal solution.')


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

plt.title('Falcon Medium Victims Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, labels=[n.id for n in node_list])
plt.xlabel("Victims and Start Node Labels")
plt.ylabel("Distance")
plt.show()