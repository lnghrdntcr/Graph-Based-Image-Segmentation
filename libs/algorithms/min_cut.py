from math import sqrt
from PIL import Image
import numpy as np
import networkx as nx
from tqdm import tqdm
from libs.linalg import *


def image_seg_min_cut(path, dist_name):
    resized_dim = (100, 100)

    dist = distances[dist_name]
    filename = path.split("/")[2].split(".")[0]
    image_path = path

    NAME = "[MIN CUT - " + dist_name + "-> " + filename + "]"

    max_dist = luminance_distance([0, 0, 0], [255, 255, 255])

    img = Image.open(image_path).convert("RGB")

    dim_x, dim_y = img.size
    img_array = np.asarray(img)

    img_graph = nx.DiGraph()
    img_graph.add_node("s")
    img_graph.add_node("t")

    for j in range(dim_x):
        for i in range(dim_y):
            img_graph.add_node((i, j))
            img_graph.add_edge("s", (i, j), capacity=dist([0, 0, 0], img_array[i][j]))
            img_graph.add_edge((i, j), "t", capacity=dist([255, 255, 255], img_array[i][j]))

    for j in tqdm(range(dim_x), desc=NAME):
        for i in range(dim_y):
            for d in [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, 1], [1, -1], [-1, 1]]:
                try:
                    next_i = i + d[0]
                    next_j = j + d[1]

                    if next_i < 0 or next_j < 0:
                        continue

                    curr_px = img_array[i][j]
                    next_px = img_array[next_i][next_j]

                    distance = dist(curr_px, next_px)

                    img_graph.add_edge((i, j), (next_i, next_j), capacity=distance)
                    img_graph.add_edge((next_i, next_j), (i, j), capacity=distance)

                except IndexError:
                    continue

    edge_set = nx.minimum_cut(img_graph, "s", "t")

    fg = edge_set[1][0]
    bg = edge_set[1][1]

    out = Image.new("RGB", img.size)

    pixels = out.load()

    for j in range(img.size[0]):
        for i in range(img.size[1]):
            pixels[j, i] = (255, 255, 255) if (i, j) in fg else (0, 0, 0)

    print(f"./segmented/min-cut/{filename}_{dist_name}.png")
    out.save(f"./segmented/min-cut/{filename}_{dist_name}.png")

    # IPython.embed()
