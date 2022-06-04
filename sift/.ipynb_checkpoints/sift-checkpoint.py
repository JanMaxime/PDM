import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def sift(test_image, baseline):
    MIN_MATCH_COUNT = 10

    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(test_image,None)
    kp2, des2 = sift.detectAndCompute(baseline,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
            
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = test_image.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        baseline = cv.polylines(baseline,[np.int32(dst)],True,255,3, cv.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
        
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    img3 = cv.drawMatches(test_image,kp1,baseline,kp2,good,None,**draw_params)
    plt.imsave( f'test2.png',img3)
    plt.imshow(img3)
    return M

def get_homography(test_image, baseline, keypoints_finder_name = "sift"):
    MIN_MATCH_COUNT = 10

    # Initiate SIFT detector
    match keypoints_finder_name:
        case "sift": keypoints_finder = cv.SIFT_create()
        case "surf": keypoints_finder = cv.xfeatures2d.SURF_create()
        case "orb" : keypoints_finder = cv.ORB_create()
        case "brisk": keypoints_finder = cv.BRISK_create()
        case _ : raise(f"Unexpected keypoints finder: {keypoints_finder_name}")

    # find the keypoints and descriptors with SIFT
    kp1, des1 = keypoints_finder.detectAndCompute(test_image,None)
    kp2, des2 = keypoints_finder.detectAndCompute(baseline,None)

    if des1 is None or des2 is None:
        return
    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
            
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        h,w = test_image.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        return M