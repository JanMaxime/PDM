import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from sift import *
import glob
import os
import json
import copy
import math
import random
import gc

def evaluate_homography(scale, gaussian_kernel, do_skeletonize, keypoints_finder_name, train_set=True):
    baseline = cv.imread('baseline.png',0)
    test_images = []
    test_points = []
    
    if(train_set):
        path = "points_full_predictions_train/*.png"
    else:
        path = "points_full_predictions_test/*.png"
    filenames = []
    for filename in glob.glob(path):
        test_images.append(cv.imread(filename,0))
        filenames.append(filename[29:-4])
        with open(f"points_transformed/{filename[29:-4]}.json","r") as json_file:
            test_points.append(json.load(json_file))
        
    baseline = cv.resize(baseline, (0,0), fx=scale, fy=scale)
    if do_skeletonize:
        baseline = (skeletonize(baseline/255)*255).astype(np.uint8)
    baseline = cv.GaussianBlur(baseline,gaussian_kernel,0)
    
    
    distances = []
    n_homography_found = 0
    
    for test_image, test_point, filename in zip(test_images, test_points, filenames):
        test_image = cv.resize(test_image, (0,0), fx=scale, fy=scale)
        if do_skeletonize:
            test_image = (skeletonize(test_image/255)*255).astype(np.uint8)
        test_image = cv.GaussianBlur(test_image,gaussian_kernel,0)
        try:
            M = get_homography(test_image, baseline, keypoints_finder_name)
        except:
            continue
        if not type(M) == np.ndarray:
            continue
        n_homography_found += 1
        
        pts = np.float32([[x*scale,y*scale] for x,y in zip(test_point["x"], test_point["y"])]).reshape(-1,1,2)
        targets = np.float32([[x*scale,y*scale] for x,y in zip(test_point["x_"], test_point["y_"])])
        p_transformed = cv.perspectiveTransform(pts,M).reshape(-1,2)
        distances.append((1/scale)*np.mean([math.sqrt((target_x - p_transformed_x)**2 + (target_y - p_transformed_y)**2) for ((target_x, target_y), (p_transformed_x, p_transformed_y)) in zip (targets, p_transformed)]))

        if not train_set:
            saved_baseline = np.copy(baseline)
            for i in range(len(targets)):
                saved_baseline = cv.circle(saved_baseline, (int(round((targets[i][0]))), int(round((targets[i][1])))), 10, (255,0,0), thickness=-1)
                saved_baseline = cv.circle(saved_baseline, (int(round((p_transformed[i][0]))), int(round((p_transformed[i][1])))), 10, (127,0,0), thickness=-1)
                #test_image = cv.circle(test_image, (int(round((pts[i][0][0]))), int(round((pts[i][0][1])))), 10, (255,0,0),-1)
            plt.imsave(f"test_results/{filename}.png", saved_baseline)
     
    mean_distance = np.mean(distances)
    if n_homography_found < 5:
        return 9999, n_homography_found
    else:
        return mean_distance, n_homography_found
    
solutions = []
keypoints_finders = ["sift", "orb", "brisk"]
for s in range(2):
    kernel = random.randint(10,80)
    if kernel % 2 == 0:
        kernel += 1
    solutions.append( ( random.uniform(0.2,0.6), kernel, random.random() >0.5, keypoints_finders[0]))
    
for e in range(2):
    ranked_solutions = []
    for s in solutions:
        ranked_solutions.append((evaluate_homography(s[0], (s[1], s[1]), s[2], s[3]),s))
    ranked_solutions.sort()
    print(f"Gen {e} best solution : {ranked_solutions[0][1]} with fitness {ranked_solutions[0][0]}")
    best_solutions = ranked_solutions[:2]
    
    scales = []
    kernels = []
    do_skeletonizes = []
    best_keypoints_finders = []
    for s in best_solutions:
        scales.append(s[1][0])
        kernels.append(s[1][1])
        do_skeletonizes.append(s[1][2])
        best_keypoints_finders.append(s[1][3])
        
        
    new_gen = []
    for i in range(20):
        scale = random.choice(scales) + random.uniform(-0.03, 0.03)
        kernel = random.choice(kernels) + random.randint(-2,2)
        if kernel % 2 == 0:
            kernel += 1
        do_skeletonize = random.choice(do_skeletonizes)
        do_skeletonize = do_skeletonize if random.random() < 0.8 else not do_skeletonize
        
        new_gen.append((scale, kernel, do_skeletonize, keypoints_finders[0]))
    solutions = new_gen