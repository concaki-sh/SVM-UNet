import numpy as np
import scipy
import skimage
import cv2


def hole_filling_per_object(image):
    """Helper function to fill holes inside individual labeled regions"""
    grow_labeled = image
    for i in np.unique(grow_labeled):
        if i == 0: continue
        filled = scipy.ndimage.morphology.binary_fill_holes(grow_labeled == i)
        grow_labeled[grow_labeled == i] = 0
        grow_labeled[filled == 1] = i
    return grow_labeled


def postprocess(result, filter_size=5, min_area=20):

    labeled, num_features = scipy.ndimage.label(result)
    for region in range(1, num_features + 1):
        area = np.sum(labeled == region)
        if area < min_area:
            result[labeled == region] = 0
    smooth = scipy.ndimage.median_filter(result, size=filter_size)
    processed = hole_filling_per_object(smooth)
    return processed




