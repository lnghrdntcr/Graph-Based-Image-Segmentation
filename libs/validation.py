from tqdm import tqdm
import numpy as np
from PIL import Image
from libs.linalg import max_euclidean, euclidean_distance
import os
from math import sqrt


def validate(debug=False, save_to_file=True):
    folders_to_validate = ["clustered", "k-cuts", "min-cut"]
    img_terminators = ["_clustered", "_k_cuts_raster", ""]

    f = None
    if save_to_file:
        f = open("./results.csv", "w+")
        f.write("image,technique,similarity,percentage_mean_distance,mean_square_error\n")

    for idx, folder in tqdm(enumerate(folders_to_validate), desc="Validation"):
        base_segmented_path = f"./segmented/{folder}"
        base_validation_path = f"./validation/{folder}"

        images = list(map(lambda x: x.split(".")[0], filter(lambda x: x.endswith(".jpg"), os.listdir("./imgs"))))
        for image in images:
            path_image_validated = f"{base_validation_path}/{image}.png"
            image_validated = np.array(Image.open(path_image_validated).convert("RGB"))
            dim_x, dim_y = len(image_validated), len(image_validated[0])
            image_validated = image_validated.reshape((1, dim_x * dim_y, 3))
            for similarity in ["euclidean_sim_norm", "luminance_sim_norm"]:
                path_image_segmented = f"{base_segmented_path}/{image}_{similarity}{img_terminators[idx]}.png"
                image_segmented = np.array(Image.open(path_image_segmented).convert("RGB")).reshape(
                    (1, dim_y * dim_x, 3))
                pixel_distance_avg = 0
                pixel_distance_stddev = 0

                for pixel_idx in range(dim_x * dim_y):
                    tmp = abs(euclidean_distance(image_validated[0][pixel_idx],
                                                 image_segmented[0][pixel_idx]) / max_euclidean)
                    pixel_distance_avg += tmp
                    pixel_distance_stddev += (tmp ** 2)

                pixel_distance_avg /= (dim_x * dim_y)
                pixel_distance_stddev /= (dim_x * dim_y)

                if pixel_distance_avg * 100 > 50:
                    # colors had been inverted
                    pixel_distance_avg = 1 - pixel_distance_avg
                    pixel_distance_stddev = 1 - pixel_distance_stddev

                if debug:
                    print(
                        f"""Validating {path_image_segmented} against {path_image_validated} ->
                            distance: {pixel_distance_avg * 100}%, stddev: {pixel_distance_stddev * 100}%""")

                if save_to_file:
                    if debug:
                        print(
                            f"Writing \n\t {image},{folder},{similarity},{pixel_distance_avg * 100},{pixel_distance_stddev * 100}")
                    f.write("{},{},{},{:3.3f},{:3.3f}\n".format(image, folder, similarity, pixel_distance_avg * 100, pixel_distance_stddev * 100))

    if save_to_file:
        f.close()
