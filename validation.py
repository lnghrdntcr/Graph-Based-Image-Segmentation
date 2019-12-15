import numpy as np
import cv2
from PIL import Image

def cv2segment(image_path, num_clusters = 2):

    filename = image_path.split("/")[2].split(".")[0]

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    vectorized = img.reshape((-1, 3))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = num_clusters
    attempts = 10
    ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape(img.shape)

    if num_clusters == 2:

        unique_colors = set()
        for el in res:
            unique_colors.add(tuple(el))

        bg = unique_colors.pop()

        tmp_image = np.ndarray((len(result_image), len(result_image[0])), dtype=(np.uint8, 3))

        for i in range(len(result_image)):
            for j in range(len(result_image[0])):
                if tuple(result_image[i][j]) == bg:
                    tmp_image[i][j] = (0, 0, 0)
                else:
                    tmp_image[i][j] = (255, 255, 255)

        out = Image.fromarray(tmp_image)
        print(f"Saving to ./validation/min-cut/{filename}.png")
        out.save(f"./validation/min-cut/{filename}.png")

        out = Image.fromarray(result_image)
        print(f"Saving to ./validation/clustered/{filename}.png")
        out.save(f"./validation/clustered/{filename}.png")

    else:
        out = Image.fromarray(result_image)
        print(f"Saving to ./validation/k-cuts/{filename}.png")
        out.save(f"./validation/k-cuts/{filename}.png")


