import cv2
import numpy as np
from draw_rectangle import draw_rectangle
import matplotlib.pyplot as plt

def chamfer_search(edge_image, template, scale, number_of_results):

    #scale and take edges of template image
    template_scaled = cv2.resize(template, (0,0), fx=scale, fy=scale)
    template_edge = cv2.Canny(template_scaled, 50, 150)
    
    #Take edges and DT of edge image
    image_edge = cv2.Canny(edge_image, 50, 150)
    edge_image_dt = cv2.distanceTransform(255 - image_edge, cv2.DIST_L2, 5)
    
    scores = cv2.filter2D(edge_image_dt, -1, template_edge)
    
    sorted_scores = np.dstack(np.unravel_index(np.argsort(scores.ravel()), scores.shape))[0]
    
    result_image = edge_image.copy()

    # Get the sorted indices of the scores
    sorted_indices = np.dstack(np.unravel_index(np.argsort(scores.ravel()), scores.shape))[0]

    boxes = []  # list to keep track of boxes' centers
    box_half_height = template_scaled.shape[0] // 2
    box_half_width = template_scaled.shape[1] // 2

    for y, x in sorted_indices:
        # Check if this point is inside any previously drawn box
        if any((abs(y - box_y) <= box_half_height) and (abs(x - box_x) <= box_half_width) for box_y, box_x in boxes):
            continue
        
        top = y - box_half_height
        left = x - box_half_width
        bottom = y + box_half_height
        right = x + box_half_width
        draw_rectangle(result_image, top, bottom, left, right)
        
        boxes.append((y, x))
        
        if len(boxes) >= number_of_results:
            break

    return scores, result_image
