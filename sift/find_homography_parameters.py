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

def evaluate_homography(scale, gaussian_kernel, do_skeletonize, crop_width, crop_height, keypoints_finder_name, train_set=True):
    baseline = cv.imread('baseline.png',0)
    baseline = baseline[crop_height:baseline.shape[0]-crop_height, crop_width:baseline.shape[1]-crop_width]
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
    bad_maps = 0
    
    for test_image, test_point, filename in zip(test_images, test_points, filenames):
        test_image = cv.resize(test_image, (0,0), fx=scale, fy=scale)
        if do_skeletonize:
            test_image = (skeletonize(test_image/255)*255).astype(np.uint8)
        test_image = cv.GaussianBlur(test_image,gaussian_kernel,0)
        try:
            M = get_homography(test_image, baseline, keypoints_finder_name)
        except:
            bad_maps +=1
            continue
        if not type(M) == np.ndarray:
            bad_maps+=1
            continue
        
        pts = np.float32([[x*scale,y*scale] for x,y in zip(test_point["x"], test_point["y"])]).reshape(-1,1,2)
        targets = np.float32([[(x-crop_width)*scale,(y-crop_height)*scale] for x,y in zip(test_point["x_"], test_point["y_"])])
        p_transformed = cv.perspectiveTransform(pts,M).reshape(-1,2)
        dst = 1/scale)*np.mean([math.sqrt((target_x - p_transformed_x)**2 + (target_y - p_transformed_y)**2) for ((target_x, target_y), (p_transformed_x, p_transformed_y)) in zip (targets, p_transformed)])
        distances.append(dst)
        if dst > 100:
            bad_maps+=1


    return bad_maps


solutions = []
keypoints_finders = ["sift", "orb", "brisk"]
num_keypoints_finder = 0
for s in range(10):
    kernel = random.randint(10,80)
    if kernel % 2 == 0:
        kernel += 1
    solutions.append( ( random.uniform(0.2,0.6), kernel, True, 0,0, keypoints_finders[num_keypoints_finder]))

print("Start")
for e in range(100):
    ranked_solutions = []
    for s in solutions:
        ranked_solutions.append((evaluate_homography(s[0], (s[1], s[1]), s[2], s[3], s[4], s[5]),s))
    ranked_solutions.sort()
    print(f"Gen {e} best solution : {ranked_solutions[0][1]} with fitness {ranked_solutions[0][0]}")
    best_solutions = ranked_solutions[:5]
    
    scales = []
    kernels = []
    do_skeletonizes = []
    crop_widths = []
    crop_heights = []
    for s in best_solutions:
        scales.append(s[1][0])
        kernels.append(s[1][1])
        do_skeletonizes.append(s[1][2])
        crop_widths.append(s[1][3])
        crop_heights.append(s[1][4])
        
        
    new_gen = []
    for i in range(10):
        scale = random.choice(scales) + random.uniform(-0.03, 0.03)
        kernel = random.choice(kernels) + random.randint(-2,2)
        if kernel % 2 == 0:
            kernel += 1
        do_skeletonize = random.choice(do_skeletonizes)
        do_skeletonize = do_skeletonize if random.random() < 0.8 else not do_skeletonize
        crop_width = random.choice(crop_widths) + random.randint(-100,100)
        crop_height = random.choice(crop_heights) + random.randint(-100,100)
        
        if crop_width > 4500:
            crop_width = 4500
        if crop_height > 3500:
            crop_height = 3500
        
        new_gen.append((scale, kernel, do_skeletonize, crop_width, crop_height, keypoints_finders[num_keypoints_finder]))
    solutions = new_gen
print("Done")
