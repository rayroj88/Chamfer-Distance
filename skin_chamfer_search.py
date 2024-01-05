import cv2
import numpy as np
import matplotlib.pyplot as plt
from draw_rectangle import draw_rectangle
from detect_skin import detect_skin

def skin_chamfer_search(color_image, edge_image, template, scale, number_of_results):
    
    # Load skin histograms from skin_hists.npy using np.load()
    negative_histogram = np.load('data/negative_histogram.npy')
    positive_histogram = np.load('data/positive_histogram.npy')
    
    #Get skin probabilities matrix
    skin_detection = detect_skin(color_image, positive_histogram, negative_histogram)
    
    #Threshold and cast to integers
    skin_mask = (skin_detection > 0.5).astype(np.uint8)

    #Scale template
    template_scaled = cv2.resize(template, (0,0), fx=scale, fy=scale) 
    
    #Binarize, find edges, and invert template image
    _, binary_template = cv2.threshold(template_scaled, 127, 255, cv2.THRESH_BINARY)
    edges_template = cv2.Canny(binary_template, 50, 150)
    edge_image_inv = (255 - edge_image).astype(np.uint8)
    
    #Distance transform on edge image and get chamfer scores
    edge_dt = cv2.distanceTransform(edge_image_inv, cv2.DIST_L2, 3)
    scores = cv2.filter2D(edge_dt, -1, edges_template, borderType=cv2.BORDER_CONSTANT)
    
    resized_skin_mask = cv2.resize(skin_mask, (scores.shape[1], scores.shape[0]))
    
    #Only keep chamfer scores in skin area
    scores = np.where(resized_skin_mask > 0, scores, np.max(scores))
    
    #Sort score indexes
    sorted_scores = np.dstack(np.unravel_index(np.argsort(scores.ravel()), scores.shape))[0]
    
    result_image = edge_image.copy()
    
    # Draw rectangles on top number_of_results
    for i in range(number_of_results):
        y, x = sorted_scores[i]
        top = y
        left = x
        bottom = top + template_scaled.shape[0]
        right = left + template_scaled.shape[1]
        draw_rectangle(result_image, top, bottom, left, right)

    return scores, result_image
