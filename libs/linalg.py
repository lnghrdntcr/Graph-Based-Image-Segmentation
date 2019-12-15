from math import sqrt


def get_deltas(x, y):
    x = list(map(int, x))
    y = list(map(int, y))
    dr = (x[0] - y[0])
    dg = (x[1] - y[1])
    db = (x[2] - y[2])
    return dr, dg, db


def luminance_distance(x, y):
    r = (int(x[0]) + int(y[0])) / 2
    dr, dg, db = get_deltas(x, y)
    return sqrt((2 + r / 256) * dr ** 2 + 4 * dg ** 2 + (2 * ((255 - r) / 256)) * db ** 2)


max_luminance = luminance_distance([0, 0, 0], [255, 255, 255])


def euclidean_distance(x, y):
    dr, dg, db = get_deltas(x, y)
    return sqrt(dr ** 2 + dg ** 2 + db ** 2)


max_euclidean = euclidean_distance([0, 0, 0], [255, 255, 255])


def euclidean_similarity(x, y):
    dr, dg, db = get_deltas(x, y)
    return max_euclidean - sqrt(dr ** 2 + dg ** 2 + db ** 2)


def euclidean_similarity_normalized(x, y):
    return euclidean_similarity(x, y) / max_euclidean


def luminance_similarity(x, y):
    r = (int(x[0]) + int(y[0])) / 2
    dr, dg, db = get_deltas(x, y)
    return max_luminance - sqrt((2 + r / 256) * dr ** 2 + 4 * dg ** 2 + (2 * ((255 - r) / 256)) * db ** 2)


def luminance_similarity_normalized(x, y):
    return luminance_similarity(x, y) / max_luminance


distances = {
    "euclidean": euclidean_distance,
    "luminance": luminance_distance,
    "euclidean_sim": euclidean_similarity,
    "luminance_sim": luminance_similarity,
    "luminance_sim_norm": luminance_similarity_normalized,
    "euclidean_sim_norm": euclidean_similarity_normalized
}
