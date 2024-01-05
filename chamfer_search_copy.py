import cv2
import numpy as np
from draw_rectangle import draw_rectangle
import matplotlib.pyplot as plt

def chamfer_search(edge_image, template, scale, number_of_results):

    #Invert the image
    edge_inv = cv2.bitwise_not(edge_image)
 
    #Resize the image to the correct scale
    resized = cv2.resize(edge_inv, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    resized_edges = cv2.Canny(resized, 50, 100)
    resized_dt = cv2.distanceTransform(255 - resized_edges, cv2.DIST_L2, 5)

    #Get template dimensions
    template_height, template_width = template.shape
    
    #create an array to store matches
    scores = np.zeros(edge_image.shape, dtype=np.float32)
    
    neighborhood_height, neighborhood_width = 10, 10 
    neighborhood_mask = np.ones((neighborhood_height, neighborhood_width), dtype=np.float32) * 9999999999999
    
    
    #Iterate through the edge image 
    for i in range(edge_image.shape[0]):
        for j in range(edge_image.shape[1]):
            
            #Calculate window for the current pixel
            window = resized_dt[i:i+template_height, j:j+template_width]          
            
            #Check if the window is in bounds
            if window.shape == template.shape:
                #Compute sum of chamfer scores in window
                chamfer_scores = np.sum(template * window)
                #add chamfer score to matches list
                scores[i:i+template_height, j:j+template_width] = chamfer_scores
    
    #make a copy of the original edge image
    result_image = edge_image.copy()
    scores_copy = scores.copy()

    #Draw rectangles around matches
    for _ in range(number_of_results):
        min_i, min_j = np.unravel_index(scores_copy.argmin(), scores.shape)
        draw_rectangle(result_image, min_i, min_i+template_height, min_j, min_j+template_width)
        
        #White out neighborhood of rectangle
        scores_copy[min_i:min_i+neighborhood_height, min_j:min_j+neighborhood_width] = neighborhood_mask
        print("coordinates after loop iteration: ")
        print(scores_copy[min_i, min_j])
        plt.figure()
        plt.imshow(result_image, cmap='gray')
        plt.show()

    
    return scores, result_image
