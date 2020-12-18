from __future__ import print_function
from ortools.linear_solver import pywraplp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import json
from random import shuffle

from environment import MapParser
import visualizer

import time

with open('data\\json\\Falcon_v1.0_Medium_sm_clean.json') as f:
    graph_json_data = json.load(f)

graph = MapParser.parse_json_map_data_new_format(graph_json_data)
start = "ew"
victim_lists = graph.victim_list.copy()




# s_size = 5
# n_size = 5

def get_distance_matrix(node_list):
    distance_matrix = np.zeros((len(node_list)+1, len(node_list)+1))
    for n1 in range(len(node_list)-1):
        for n2 in range(n1, len(node_list)):
            length = nx.dijkstra_path_length(graph, node_list[n1], node_list[n2])
            distance_matrix[n1][n2] = length
    distance_matrix = distance_matrix + distance_matrix.transpose()
    return distance_matrix.tolist()

# node_list = [graph[start]] + victim_lists


time_list = []
shuffle(victim_lists)
max_size = 19
for size in range(1, max_size):

    node_list = [graph[start]] + victim_lists[:size]

    s_size = len(node_list)
    n_size = len(node_list)

    # s_size = 4
    # n_size = 4

    solver = pywraplp.Solver.CreateSolver('SCIP')
    x = {}
    z = {}
    for s in range(s_size):
        for i in range(n_size):
            x[f"{s}_{i}"] = solver.BoolVar(f"x[{s}_{i}]")

    for s in range(s_size-1):
        for i in range(n_size):
            for j in range(n_size):
                z[f"{s}_{i}_{j}"] = solver.BoolVar(f"z[{s}_{i}_{j}]")
                solver.Add(z[f"{s}_{i}_{j}"] <= x[f"{s}_{i}"])
                solver.Add(z[f"{s}_{i}_{j}"] <= x[f"{s+1}_{j}"])
                solver.Add(z[f"{s}_{i}_{j}"] >= x[f"{s}_{i}"] + x[f"{s+1}_{j}"] - 1)

    for s in range(s_size):
        constraint_expr = [x[f"{s}_{i}"] for i in range(n_size)]
        solver.Add(sum(constraint_expr) == 1)
    #
    for i in range(n_size):
        constraint_expr = [x[f"{s}_{i}"] for s in range(s_size)]
        solver.Add(sum(constraint_expr) == 1)
    #



    solver.Add(x["0_0"] == 1)

    distance_matrix = get_distance_matrix(node_list)

    # distance_matrix = [
    #     [0,1,2,4],
    #     [1,0,2,1],
    #     [2,2,0,1],
    #     [4,1,1,0],
    # ]

    # distance_matrix = [
    #     [0,2,1,3,0],
    #     [2,0,2,1,0],
    #     [1,2,0,2,0],
    #     [3,1,2,0,0],
    #     [0,0,0,0,0]
    # ]

    obj_expr = [distance_matrix[i][j] * z[f"{s}_{i}_{j}"] for s in range(s_size-1) for i in range(n_size) for j in range(n_size)]
    solver.Minimize(solver.Sum(obj_expr))

    print("here")

    solver.SetTimeLimit(1000000)
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print('Objective value =', solver.Objective().Value())
        # for s in range(s_size):
        #     for i in range(n_size):
        #         print(int(x[f"{s}_{i}"].solution_value()), end=" ")
        #     print()


        # curr = 0
        # full_path = []
        # victim_path = []
        # for s in range(s_size-1):
        #     for j in range(n_size):
        #         if int(x[f"{s}_{curr}_{j}"].solution_value()) == 1:
        #             curr = j
        #             victim_path.append(node_list[j].id)
        # for i in range(len(victim_path)-1):
        #     full_path += list(map(lambda x:x.id, nx.dijkstra_path(graph, graph[victim_path[i]], graph[victim_path[i+1]])))
        # visualizer.animate_graph_training_json(full_path, graph_json_data, with_save="TSP_MIP_MIN")

        print('Problem solved in %f milliseconds' % solver.wall_time())
        print('Problem solved in %d iterations' % solver.iterations())
        print('Problem solved in %d branch-and-bound nodes' % solver.nodes())
        # time_list.append(solver.wall_time() / 1000)

    else:
        print('The problem does not have an optimal solution.')

# plt.plot(range(1, max_size), time_list)
# plt.xlabel("Number of victims")
# plt.ylabel("Time in seconds")
# plt.show()
