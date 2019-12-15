import os
from libs.algorithms.clustering import image_seg_clustering
from libs.algorithms.k_cuts import image_seg_k_cuts
from libs.algorithms.min_cut import image_seg_min_cut
from multiprocessing import Process
from time import time
from validation import cv2segment
from libs.validation import validate


def print_banner():
    print("""
     _____                             _____                                 _        _   _                         
    |_   _|                           /  ___|                               | |      | | (_)                        
      | | _ __ ___   __ _  __ _  ___  \ `--.  ___  __ _ _ __ ___   ___ _ __ | |_ __ _| |_ _  ___  _ __              
      | || '_ ` _ \ / _` |/ _` |/ _ \  `--. \/ _ \/ _` | '_ ` _ \ / _ \ '_ \| __/ _` | __| |/ _ \| '_ \             
     _| || | | | | | (_| | (_| |  __/ /\__/ /  __/ (_| | | | | | |  __/ | | | || (_| | |_| | (_) | | | |            
     \___/_| |_| |_|\__,_|\__, |\___| \____/ \___|\__, |_| |_| |_|\___|_| |_|\__\__,_|\__|_|\___/|_| |_|            
                           __/ |                   __/ |                                                            
    ___  ____             _____       _         __|___/ ___      _ _   _                            _____       _   
    |  \/  (_)           /  __ \     | |       / / |  \/  |     | | | (_)                          /  __ \     | |  
    | .  . |_ _ __ ______| /  \/_   _| |_     / /  | .  . |_   _| | |_ ___      ____ _ _   _ ______| /  \/_   _| |_ 
    | |\/| | | '_ \______| |   | | | | __|   / /   | |\/| | | | | | __| \ \ /\ / / _` | | | |______| |   | | | | __|
    | |  | | | | | |     | \__/\ |_| | |_   / /    | |  | | |_| | | |_| |\ V  V / (_| | |_| |      | \__/\ |_| | |_ 
    \_|  |_/_|_| |_|      \____/\__,_|\__| /_/     \_|  |_/\__,_|_|\__|_| \_/\_/ \__,_|\__, |       \____/\__,_|\__|
                                                                                        __/ |                       
                                                                                       |___/
        """)


TEST_SUITE = {
    "clustering": image_seg_clustering,
    "min_cut": image_seg_min_cut,
    "k_cuts": image_seg_k_cuts,
}


def run_tests():
    t_begin = time()
    processes = []
    for test in TEST_SUITE.keys():
        images = list(map(lambda y: "./imgs/" + y, list(filter(lambda x: x.endswith(".jpg"), os.listdir("./imgs")))))
        print(f"Performing test on {images} using {test}")
        for image in images:
            for similarity in ["euclidean_sim_norm", "luminance_sim_norm"]:
                tx = Process(target=TEST_SUITE[test], args=(image, similarity))
                tx.start()
                processes.append(tx)

        for process in processes:
            process.join()

    t_end = time()

    print(f"Computation took {t_end - t_begin}s")


def generate_golden_images():
    images = list(map(lambda y: "./imgs/" + y, list(filter(lambda x: x.endswith(".jpg"), os.listdir("./imgs")))))
    print(f"Performing opencv segmentation on {images}")
    for image in images:
        cv2segment(image)
        cv2segment(image, num_clusters=4)


if __name__ == "__main__":
    print_banner()
    run_tests()
    generate_golden_images()
    validate(save_to_file=True)
