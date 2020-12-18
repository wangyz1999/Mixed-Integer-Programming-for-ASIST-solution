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

# vic_pos = {}
# green_vic = []
# yellow_vic = []
# for obj in graph_json_data["objects"]:
#     if obj["type"] == "green_victim":
#         green_vic.append({"id":obj["id"], "x":obj["bounds"]["coordinates"][0]["x"], "z":obj["bounds"]["coordinates"][0]["z"]})
#     if obj["type"] == "yellow_victim":
#         yellow_vic.append({"id":obj["id"], "x":obj["bounds"]["coordinates"][0]["x"], "z":obj["bounds"]["coordinates"][0]["z"]})
#     vic_pos["victims"] = green_vic + yellow_vic
#
# with open('victims_medium.json', 'w') as f:
#     json.dump(vic_pos, f)

# s_size = 5
# n_size = 5

def get_clusters(graph):
    cluster_rooms = dict()
    for r in graph.room_list:
        room_id = r.id.split("_")[0]
        if len(r.victim_list) > 0:
            cluster_rooms[room_id] = []
    for r in graph.room_list:
        room_id = r.id.split("_")[0]
        for v in r.victim_list:
            if v not in cluster_rooms[room_id]:
                cluster_rooms[room_id].append(v)
    return cluster_rooms





print(get_clusters(graph))



# def get_distance_matrix(node_list):
#     distance_matrix = np.zeros((len(node_list)+1, len(node_list)+1))
#     for n1 in range(len(node_list)-1):
#         for n2 in range(n1, len(node_list)):
#             length = nx.dijkstra_path_length(graph, node_list[n1], node_list[n2])
#             distance_matrix[n1][n2] = length
#     distance_matrix = distance_matrix + distance_matrix.transpose()
#     return distance_matrix.tolist()
#
# # node_list = [graph[start]] + victim_lists
#
#
# time_list = []
# shuffle(victim_lists)
# max_size = 19
# for size in range(1, max_size):
#     node_list = [graph[start]] + victim_lists[:size]
#
#     s_size = len(node_list) + 1
#     n_size = len(node_list) + 1
#
#     solver = pywraplp.Solver.CreateSolver('SCIP')
#     x = {}
#     for s in range(s_size):
#         for i in range(n_size):
#             for j in range(n_size):
#                 x[f"{s}_{i}_{j}"] = solver.BoolVar(f"x[{s}_{i}_{j}]")
#
#
#     for s in range(s_size):
#         constraint_expr = [x[f"{s}_{i}_{j}"] for i in range(n_size) for j in range(n_size)]
#         solver.Add(sum(constraint_expr) == 1)
#     #
#     for i in range(n_size):
#         constraint_expr = []
#         for s in range(s_size):
#             for j in range(n_size):
#                 if i != j:
#                     constraint_expr.append(x[f"{s}_{i}_{j}"])
#         solver.Add(sum(constraint_expr) == 1)
#     #
#     for j in range(n_size):
#         constraint_expr = []
#         for s in range(s_size):
#             for i in range(n_size):
#                 if i != j:
#                     constraint_expr.append(x[f"{s}_{i}_{j}"])
#         solver.Add(sum(constraint_expr) == 1)
#
#     # for s in range(s_size-1):
#     #     for i in range(n_size):
#     #         for j in range(n_size):
#     #             constraint_expr = []
#     #             for k in range(n_size):
#     #                 constraint_expr.append(x[f"{s+1}_{j}_{k}"])
#     #             solver.Add(x[f"{s}_{i}_{j}"] - sum(constraint_expr) == 0)
#
#     for s in range(s_size-1):
#         for j in range(n_size):
#             prev_node = []
#             next_node = []
#             for i in range(n_size):
#                 prev_node.append(x[f"{s}_{i}_{j}"])
#             for k in range(n_size):
#                 next_node.append(x[f"{s+1}_{j}_{k}"])
#             solver.Add(sum(prev_node) - sum(next_node) == 0)
#
#     solver.Add(x[f"0_{len(node_list)}_0"] == 1)
#
#     distance_matrix = get_distance_matrix(node_list)
#
#     # distance_matrix = [
#     #     [0,1,2,4,0],
#     #     [1,0,2,1,0],
#     #     [2,2,0,1,0],
#     #     [4,1,1,0,0],
#     #     [0,0,0,0,0]
#     # ]
#
#     # distance_matrix = [
#     #     [0,2,1,3,0],
#     #     [2,0,2,1,0],
#     #     [1,2,0,2,0],
#     #     [3,1,2,0,0],
#     #     [0,0,0,0,0]
#     # ]
#
#     obj_expr = [distance_matrix[i][j] * x[f"{s}_{i}_{j}"] for s in range(s_size) for i in range(n_size) for j in range(n_size)]
#     solver.Minimize(solver.Sum(obj_expr))
#
#     print("here")
#
#     solver.SetTimeLimit(1000000)
#     status = solver.Solve()
#
#     if status == pywraplp.Solver.OPTIMAL:
#         # print('Objective value =', solver.Objective().Value())
#         # for s in range(s_size):
#         #     print(f"s={s}")
#         #     for i in range(n_size):
#         #         for j in range(n_size):
#         #             print(int(x[f"{s}_{i}_{j}"].solution_value()), end=" ")
#         #         print()
#         #     print()
#
#
#         # curr = 0
#         # full_path = []
#         # victim_path = []
#         # for s in range(s_size-1):
#         #     for j in range(n_size):
#         #         if int(x[f"{s}_{curr}_{j}"].solution_value()) == 1:
#         #             curr = j
#         #             victim_path.append(node_list[j].id)
#         # for i in range(len(victim_path)-1):
#         #     full_path += list(map(lambda x:x.id, nx.dijkstra_path(graph, graph[victim_path[i]], graph[victim_path[i+1]])))
#         # visualizer.animate_graph_training_json(full_path, graph_json_data, with_save="TSP_MIP_MIN")
#
#         print('Problem solved in %f milliseconds' % solver.wall_time())
#         print('Problem solved in %d iterations' % solver.iterations())
#         print('Problem solved in %d branch-and-bound nodes' % solver.nodes())
#         time_list.append(solver.wall_time() / 1000)
#
#     else:
#         print('The problem does not have an optimal solution.')
#
# plt.plot(range(1, max_size), time_list)
# plt.xlabel("Number of victims")
# plt.ylabel("Time in seconds")
# plt.show()
