import string
import networkx as nx
import copy
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from libs.linalg import *


def merge_nodes(G: nx.Graph, node_list):
    new_graph = G.copy()
    new_graph.add_node("s_x")

    #print(node_list)
    # for all nodes that point to any of node_list
    edges = list(new_graph.edges(data="capacity", nbunch=node_list))

    for n1, n2, capacity in edges:
        if n2 in node_list:
            # if edge exists
            if new_graph.has_edge(n1, "s_x"):
                new_graph[n1]["s_x"]["capacity"] += capacity
            else:
                new_graph.add_edge(n1, "s_x", capacity=capacity)
        elif n1 in node_list:
            if new_graph.has_edge(n2, "s_x"):
                new_graph[n2]["s_x"]["capacity"] += capacity
            else:
                new_graph.add_edge(n2, "s_x", capacity=capacity)

    for n1, n2, capacity, in new_graph.edges(data="capacity", nbunch=["s_x"]):
        new_graph[n1][n2]["capacity"] = capacity / len(node_list)

    for n in node_list:
        new_graph.remove_node(n)

    return new_graph


def create_img_graph(img, sectors, using=nx.Graph, dist=euclidean_distance, directed=False):
    dim_x, dim_y = img.size
    img_array = np.asarray(img)
    graph_generator = using

    img_graph = graph_generator()
    # img_graph.add_node("s")
    # img_graph.add_node("t")

    for j in range(dim_x):
        for i in range(dim_y):
            img_graph.add_node((i, j))
            for idx, c in enumerate(sectors):
                img_graph.add_edge(f"s_{idx}", (i, j), capacity=dist(c, img_array[i][j]))
                if directed:
                    img_graph.add_edge((i, j), f"s_{idx}", capacity=dist(c, img_array[i][j]))

    for j in tqdm(range(dim_x)):
        for i in range(dim_y):
            for d in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                try:
                    next_i = i + d[0]
                    next_j = j + d[1]

                    if next_i < 0 or next_j < 0:
                        continue

                    curr_px = img_array[i][j]
                    next_px = img_array[next_i][next_j]

                    distance = dist(curr_px, next_px)

                    img_graph.add_edge((i, j), (next_i, next_j), capacity=distance)
                    if directed:
                        img_graph.add_edge((next_i, next_j), (i, j), capacity=distance)

                except IndexError:
                    continue

    return img_graph
