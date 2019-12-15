from math import sqrt
from PIL import Image
import numpy as np
import networkx as nx
from tqdm import tqdm
import scipy.cluster
from libs.linalg import *
import random


def image_seg_clustering(path, dist_name):
    resized_dim = (100, 100)

    dist = distances[dist_name]
    filename = path.split("/")[2].split(".")[0]
    image_path = path

    NAME = "[COLOR CLUSTERING - " + dist_name + " -> " + filename + "]"

    max_dist = luminance_distance([0, 0, 0], [255, 255, 255])

    img = Image.open(image_path).convert("RGB")

    # Clustering
    img_clustered = img.resize(resized_dim)
    img_clustered = np.asarray(img_clustered)
    shape = img_clustered.shape
    img_clustered = img_clustered.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    codes, _ = scipy.cluster.vq.kmeans(img_clustered, 2)

    main_fg = list(map(int, codes[0]))
    main_bg = list(map(int, codes[1]))

    print(f"{NAME} Most common colors {main_bg}, {main_fg}")

    dim_x, dim_y = img.size
    img_array = np.asarray(img)

    img_graph = nx.Graph()
    img_graph.add_node("s")
    img_graph.add_node("t")

    for j in range(dim_x):
        for i in range(dim_y):
            img_graph.add_node((i, j))
            img_graph.add_edge("s", (i, j), capacity=dist(main_fg, img_array[i][j]))
            img_graph.add_edge((i, j), "t", capacity=dist(main_bg, img_array[i][j]))

    for j in tqdm(range(dim_x), desc=NAME):
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
                    # img_graph.add_edge((next_i, next_j), (i, j), capacity=distance)

                except IndexError:
                    continue

    # print(f"{NAME} Computing max_flow")
    # edge_set = nx.minimum_edge_cut(img_graph)
    edge_set = nx.minimum_cut(img_graph, "s", "t")

    bg = edge_set[1][0]
    fg = edge_set[1][1]

    out = Image.new("RGB", img.size)

    pixels = out.load()

    for j in range(img.size[0]):
        for i in range(img.size[1]):
            pixels[j, i] = tuple(main_fg) if (i, j) in fg else tuple(main_bg)

    print(f"Saving into ./segmented/clustered/{filename}_{dist_name}_clustered.png")
    out.save(f"./segmented/clustered/{filename}_{dist_name}_clustered.png")

    # IPython.embed()
