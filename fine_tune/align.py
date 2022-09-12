import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import  Image, ImageDraw
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import NearestNeighbors
import cv2 as cv

def align(im1, imr, neighbor_radius, threshold=0):
    
    ## Get the key points of the reference map
    MAX_MATCHES = 500
    GOOD_MATCH_PERCENT = 0.15

    points0 = []
    orb = cv.ORB_create(MAX_MATCHES)
    r_keypoints, r_descriptors = orb.detectAndCompute(imr, None)

    for point in r_keypoints:
        points0.append(point.pt)
        
    ## Get the key points of the target map
    keypoints1, descriptors1 = orb.detectAndCompute(im1, None)

    points1 = []
    for point in keypoints1:
        points1.append(point.pt)
    
    ## Find the neighboring keypoints indices of each target keypoint
    neigh = NearestNeighbors(radius=neighbor_radius)
    neigh.fit(points0)
    neighboring_indicies = neigh.radius_neighbors(points1, return_distance=False)
    
    ## Sort the indices according to the descriptor similarity
    sorted_neighboring_indicies = np.copy(neighboring_indicies)
    similarities = np.copy(neighboring_indicies)

    for i, neighboring_indicie in enumerate(neighboring_indicies):
        if not np.any(neighboring_indicie):
            continue
        similarity = np.dot(r_descriptors[neighboring_indicie], descriptors1[i])
        the_norm = np.linalg.norm(r_descriptors[neighboring_indicie], axis=1)
        the_norm = the_norm * np.linalg.norm(descriptors1[i])
        similarity = np.divide(similarity, the_norm)
        sorted_neighboring_indicies[i] = np.flipud(neighboring_indicie[np.argsort(similarity)])
        similarities[i] = np.flipud(similarity)
        
    ## Prepare the data for cross check
    neigh1 = NearestNeighbors(radius=neighbor_radius)
    neigh1.fit(points1)
    c_neighboring_indicies = neigh1.radius_neighbors(points0, return_distance=False)

    c_sorted_neighboring_indicies = np.copy(c_neighboring_indicies)
    c_similarities = np.copy(c_neighboring_indicies)

    for i, neighboring_indicie in enumerate(c_neighboring_indicies):
        if not np.any(neighboring_indicie):
            continue        
        similarity = np.dot(descriptors1[neighboring_indicie], r_descriptors[i])
        the_norm = np.linalg.norm(descriptors1[neighboring_indicie], axis=1)
        the_norm = the_norm * np.linalg.norm(r_descriptors[i])
        similarity = np.divide(similarity, the_norm)
        c_sorted_neighboring_indicies[i] = np.flipud(neighboring_indicie[np.argsort(similarity)])
        c_similarities[i] = np.flipud(similarity)
    
    ## Perform cross check and output the matches according to the performs
    match_similarities = []
    match_indecies_pairs = []

    for i, neighboring_indicie in enumerate(neighboring_indicies):
        if not np.any(neighboring_indicie):
            continue
        best_match = neighboring_indicie[0]
        if not np.any(c_sorted_neighboring_indicies[best_match]):
            continue
        if i == c_sorted_neighboring_indicies[best_match][0]:
            match_indecies_pairs.append(tuple([best_match, i])) # (reference_map_point, other_map_point)
            match_similarities.append(similarities[i][0])

    sorted_match_similarities = np.flipud(np.sort(match_similarities))
    sorted_match_indecies_pairs = np.flipud(np.array(match_indecies_pairs)[np.argsort(match_similarities)])
    
    # Allign the map
    num_matches = np.sum(sorted_match_similarities > threshold)

    reference_points = []
    process_points = []

    for i in range(num_matches):
        reference_points.append(points0[sorted_match_indecies_pairs[i][0]])
        process_points.append(points1[sorted_match_indecies_pairs[i][1]])
        
    h, mask = cv.findHomography(np.array(process_points), np.array(reference_points), cv.RANSAC)

    height, width = imr.shape
    alligned = cv.warpPerspective(im1, h, (width, height))
    return alligned,h

def align_flexible_radius(map2, map1, reference_radius, index=0, step_size=5, save_each_map=False, save_best_map=False, draw_figure=False, save_figure=False, map_name="test"):
    radius = np.arange(-3, 4, 1, dtype=int) * int(step_size) + int(reference_radius)
    alligned_map2 = []
    sum_of_none_zero_entries = []
    Ms = []
    for i in radius:
        alligned_map2_i,M = align(map2, map1, i)
        alligned_map2.append(alligned_map2_i)
        Ms.append(M)
        sum_of_none_zero_entries.append(np.count_nonzero(alligned_map2_i + map1))
        if save_each_map:
            to_draw_1 = map1 + alligned_map2_i
            img_alligned_map2_i = Image.fromarray(to_draw_1[800:6000, 1500:8500])
            img_alligned_map2_i.save("map{}_radius{}.png".format(index, i))
            
    best_alligned = alligned_map2[np.argmin(sum_of_none_zero_entries)]
    best_radius = radius[np.argmin(sum_of_none_zero_entries)]
    best_M = Ms[np.argmin(sum_of_none_zero_entries)]
    
    if save_best_map:
        to_draw_2 = map1 + best_alligned
        best_alligned_image = Image.fromarray(to_draw_2[800:6000, 1500:8500])
        #best_alligned_image.save()
        plt.imsave(f"aligned_orb/{map_name}.png", best_alligned_image)
    
    if draw_figure:
        fig, ax = plt.subplots()
        ax.plot(radius, sum_of_none_zero_entries)
        plt.xlabel('Search radius(pixels)')
        plt.ylabel('Number of active pixels')
        plt.title('Number of active pixels versus search radius')
        if save_figure:
            plt.savefig("similarity_versus_radius_map_{}.png".format(index))
        plt.close(fig)
    
    return best_alligned, best_radius, M