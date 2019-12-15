from PIL import Image
from libs.graph_utils import *
from libs.image_utils import *
from functools import reduce


def image_seg_k_cuts(path, dist_name):
    n_clusters = 5
    dist = distances[dist_name]
    filename = path.split("/")[2].split(".")[0]
    image_path = path

    NAME = "[K CUTS - " + dist_name + " -> " + filename + "]"

    cuts = []
    costs = []

    max_dist = luminance_distance([0, 0, 0], [255, 255, 255])

    img = Image.open(image_path).convert("RGB")

    sectors = k_colors(img, n_clusters)
    img_graph = create_img_graph(img, sectors, using=nx.Graph, dist=dist)

    for i in tqdm(range(n_clusters), desc=NAME):
        # print("\nMerging nodes")
        ng = merge_nodes(img_graph, [f"s_{j}" for j in range(n_clusters) if j != i])
        # print("\nExecuting minimum_cut")
        cost, cut = nx.minimum_cut(ng, f"s_{i}", "s_x")
        x, y = cut
        costs.append(cost)
        cut = set(nx.algorithms.edge_boundary(ng, x, y))
        ncut = set()
        for (i1, j1) in cut:
            if i1 == "s_x":
                for sink in [f"s_{j}" for j in range(n_clusters) if j != i]:
                    ncut.add((sink, j1))
            elif j1 == "s_x":
                for sink in [f"s_{j}" for j in range(n_clusters) if j != i]:
                    ncut.add((i1, sink))
            else:
                ncut.add((i1, j1))
        cuts.append(ncut)

    idx_max = costs.index(max(costs))
    cuts.pop(idx_max)

    final_cut = set()

    for s in cuts:
        final_cut = final_cut.union(s)

    img_graph.remove_edges_from(final_cut)
    conn_comp = list(nx.connected_components(img_graph))

    pixel_clusters = sorted(conn_comp, reverse=True, key=len)[:n_clusters]

    out = Image.new("RGB", img.size)

    pixels = out.load()

    img_pixels = np.asarray(img)
    avgs = []
    for comp in conn_comp:
        r = 0
        b = 0
        g = 0
        for pix in comp:
            if pix[0] != "s":
                r1, g1, b1 = img_pixels[pix[0]][pix[1]]
                r += r1
                b += b1
                g += g1
        avgs.append(tuple(map(lambda x: int(x / len(comp)), (r, g, b))))

    for j in range(img.size[0]):
        for i in range(img.size[1]):
            color = (0, 0, 0)
            for idx, pixel_cluster in enumerate(conn_comp):
                if (i, j) in pixel_cluster:
                    color = avgs[idx]
            pixels[j, i] = tuple(color)

    # for j in range(img.size[0]):
    #     for i in range(img.size[1]):
    #         color = (0, 0, 0)
    #         for pixel_cluster in sorted(conn_comp, reverse=True, key=len)[n_clusters:]:
    #             if (i,j) in pixel_cluster:
    #                 color = (0,255,0)
    #         pixels[j, i] = tuple(color)

    print(f"Saving into ./segmented/k-cuts/{filename}_{dist_name}_k_cuts_raster.png")
    out.save(f"./segmented/k-cuts/{filename}_{dist_name}_k_cuts_raster.png")

    # IPython.embed()
