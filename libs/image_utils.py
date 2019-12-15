import numpy as np
import scipy
import scipy.cluster

import binascii

# print('finding clusters')
# codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
# print('cluster centres:\n', codes)
#
# vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
# counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences
#
# index_max = scipy.argmax(counts)                    # find most frequent
# peak = codes[index_max]
# colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
# print('most frequent is %s (#%s)' % (peak, colour))

def k_colors(img, clusters, resized_dim=(100, 100)):
    img_clustered = img.resize(resized_dim)
    img_clustered = np.asarray(img_clustered)
    shape = img_clustered.shape
    scipy.linalg
    img_clustered = img_clustered.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
    codes, _ = scipy.cluster.vq.kmeans(img_clustered, clusters)
    vecs, dist = scipy.cluster.vq.vq(img_clustered, codes)         # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

    count_map = {}
    for idx, count in enumerate(counts):
        count_map[idx] = count

    sorted_counts = list(sorted([i for i in count_map.items()], reverse=True, key=lambda x: x[1]))

    final_codes = []
    for el in sorted_counts:
        final_codes.append(codes[el[0]])

    return list(map(lambda x: list(map(int, x)), final_codes))
