from __future__ import print_function
from ortools.linear_solver import pywraplp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import json
from random import shuffle

from environment import MapParser
import visualizer
from graph.Nodes import VictimType

import time

def get_distance_matrix(graph, node_list):
    distance_matrix = np.zeros((len(node_list), len(node_list)))
    for n1 in range(len(node_list)-1):
        for n2 in range(n1, len(node_list)):
            length = nx.dijkstra_path_length(graph, node_list[n1], node_list[n2])
            distance_matrix[n1][n2] = length
    distance_matrix = distance_matrix + distance_matrix.transpose()
    return distance_matrix.tolist()

def sep_yellow_green_victim_list(victim_list):
    yellow_victims = []
    green_victims = []
    for v in victim_list:
        if v.victim_type == VictimType.Yellow:
            yellow_victims.append(v)
        elif v.victim_type == VictimType.Green:
            green_victims.append(v)
    return yellow_victims, green_victims

def initialize(graph_json_data):
    graph = MapParser.parse_json_map_data_new_format(graph_json_data)
    start = "ew"
    victim_list = graph.victim_list.copy()
    yellow_victims, green_victims = sep_yellow_green_victim_list(victim_list)
    # node_list = [graph[start]] + yellow_victims + green_victims
    victim_list = yellow_victims + green_victims
    distance_matrix = get_distance_matrix(graph, victim_list)
    data = {
        "node_list": victim_list,
        "distance_matrix": distance_matrix,
        "num_yellow": len(yellow_victims),
        "num_green": len(green_victims),
        "num_all_nodes": len(victim_list)
    }
    return data

def toy_example_1():
    distance_matrix = [
        [0, 1, 4, 2, 0],
        [1, 0, 1, 2, 0],
        [4, 1, 0, 1, 0],
        [2, 2, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    data = {
        "distance_matrix": distance_matrix,
        "num_yellow": 2,
        "num_green": 2,
        "num_all_nodes": 4
    }
    return data

def toy_example_2():
    distance_matrix = [
        [0, 1, 4, 2, 3, 4, 3, 0],
        [1, 0, 1, 2, 5, 1, 3, 0],
        [4, 1, 0, 1, 2, 6, 2, 0],
        [2, 2, 1, 0, 3, 5, 2, 0],
        [3, 5, 2, 3, 0, 4, 1, 0],
        [4, 1, 6, 5, 4, 0, 7, 0],
        [3, 3, 2, 2, 1, 7, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]
    data = {
        "distance_matrix": distance_matrix,
        "num_yellow": 2,
        "num_green": 2,
        "num_all_nodes": 4
    }
    return data

def mip_solve(data):
    n = data["num_all_nodes"]
    solver = pywraplp.Solver.CreateSolver('CBC')
    M = 10000
    X = {}
    for s in range(n):
        for i in range(n):
            for j in range(n):
                X[f"{s}_{i}_{j}"] = solver.BoolVar(f"X[{s}_{i}_{j}]")

    for s in range(n):
        constraint_expr = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    constraint_expr.append(X[f"{s}_{i}_{j}"])
        solver.Add(sum(constraint_expr) == 1)

    for i in range(n):
        constraint_expr = []
        for s in range(n):
            for j in range(n):
                if i != j:
                    constraint_expr.append(X[f"{s}_{i}_{j}"])
        solver.Add(sum(constraint_expr) == 1)

    for j in range(n):
        constraint_expr = []
        for s in range(n):
            for i in range(n):
                if i != j:
                    constraint_expr.append(X[f"{s}_{i}_{j}"])
        solver.Add(sum(constraint_expr) == 1)

    for s in range(n-1):
        for j in range(n):
            prev_node = []
            next_node = []
            for i in range(n):
                prev_node.append(X[f"{s}_{i}_{j}"])
            for k in range(n):
                next_node.append(X[f"{s+1}_{j}_{k}"])
            solver.Add(sum(prev_node) - sum(next_node) == 0)

    Y = {}
    for s in range(n):
        for j in range(n):
            Y[f"{s}_{j}"] = solver.BoolVar(f"Y[{s}_{j}]")
    for s in range(n):
        for j in range(n):
            constraint_expr = []
            for i in range(n):
                constraint_expr.append(X[f"{s}_{i}_{j}"])
            solver.Add(Y[f"{s}_{j}"] == sum(constraint_expr))


    D = {}
    for s in range(n):
        for j in range(n):
            D[f"{s}_{j}"] = solver.IntVar(0, solver.infinity(), f'D[{s}_{j}]')
    for s in range(n):
        for j in range(n):
            constraint_expr = []
            for i in range(n):
                constraint_expr.append(X[f"{s}_{i}_{j}"] * data["distance_matrix"][i][j])
            D[f"{s}_{j}"] = sum(constraint_expr)

    ST = {}
    for s in range(n):
        ST[f"{s}"] = solver.IntVar(0, solver.infinity(), f'ST[{s}]')
    for _s in range(n):
        constraint_expr = []
        for s in range(_s):
            for i in range(n):
                constraint_expr.append(D[f"{s}_{i}"])
        solver.Add(ST[f"{_s}"] == sum(constraint_expr))

    T = {}
    for i in range(n):
        T[f"{i}"] = solver.IntVar(0, solver.infinity(), f'T[{i}]')
    for s in range(1, n):
        for i in range(n):
            solver.Add(T[f"{i}"] >= ST[f"{s-1}"] + D[f"{s}_{i}"] - M*(1 - Y[f"{s}_{i}"]))

    Threshold_yellow = 5 * 60 / 5.6
    Threshold_green = 10 * 60 / 5.6

    V = {}
    for i in range(n):
        constraint_expr = []
        for s in range(n):
            constraint_expr.append(Y[f"{s}_{i}"])
        V[f"{i}"] = sum(constraint_expr)

    for i in range(data["num_yellow"]):
        solver.Add(T[f"{i}"] - M*(1 - V[f"{i}"]) <= Threshold_yellow)

    for i in range(data["num_yellow"], data["num_all_nodes"]):
        solver.Add(T[f"{i}"] - M*(1 - V[f"{i}"]) <= Threshold_green)

    Reward_yellow = 30
    Reward_green = 10

    obj_expr_1 = sum([T[f"{i}"] for i in range(n)])
    obj_expr_2 = sum([Reward_yellow * V[f"{i}"] for i in range(data["num_yellow"])])
    obj_expr_3 = sum([Reward_green * V[f"{i}"] for i in range(data["num_yellow"], data["num_all_nodes"])])
    solver.Minimize(obj_expr_1 - obj_expr_2 - obj_expr_3)

    solver.SetTimeLimit(100000)
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print('Objective value =', solver.Objective().Value())
        print('Problem solved in %f milliseconds' % solver.wall_time())
        print('Problem solved in %d iterations' % solver.iterations())
        print('Problem solved in %d branch-and-bound nodes' % solver.nodes())

        print("below print for X")
        for s in range(n):
            print(f"s={s}")
            for i in range(n):
                for j in range(n):
                    print(int(X[f"{s}_{i}_{j}"].solution_value()), end=" ")
                print()
            print()

        print("below print for Y")
        for s in range(n):
            for i in range(n):
                print(int(Y[f"{s}_{i}"].solution_value()), end=" ")
            print()
        print()

        print("below print for V")
        for i in range(n):
            print(int(V[f"{i}"].solution_value()), end=" ")
        print("\n")

        print("below print for T")
        for i in range(n):
            print(int(T[f"{i}"].solution_value()), end=" ")
        print("\n")

        print("below print for ST")
        for s in range(n):
            print(int(ST[f"{s}"].solution_value()), end=" ")
        print("\n")

        print("below print for D")
        for s in range(n):
            for i in range(n):
                print(int(D[f"{s}_{i}"].solution_value()), end=" ")
            print()
        print()
    else:
        print('The problem does not have an optimal solution.')

if __name__ == "__main__":
    with open('data\\json\\Falcon_v1.0_Medium_sm_clean.json') as f:
        graph_json_data = json.load(f)

    # data = initialize(graph_json_data)
    data = toy_example_1()
    # data = toy_example_2()

    mip_solve(data)


