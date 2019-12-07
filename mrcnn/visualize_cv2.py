"""
Mask R-CNN
Display and Visualization Functions for the python-opencv.
"""
import os
import sys
import random
import itertools
import colorsys

import numpy as np 
import cv2

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)

    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image

def mask_gray(image, mask):
    # Convert Mask to gray image
    image[:, :] = np.where(mask == 1,
                            image[:, :]*255,
                            image[:, :]-1)
    return image


def cv2_save_instances(image, boxes, masks, class_ids, class_names, scores=None, 
                      output_image_path=None, title="", save_image=False, 
                      show=False, show_mask=True, show_bbox=True, 
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    nCounts = boxes.shape[0]
    if not nCounts:
        print("\n*** No instance to display ***\n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
    
    # Generate random colors
    colors = colors or random_colors(nCounts)

    mask_image = np.zeros(image.shape, dtype=np.uint8)
    mask_image[:, :, 0] = image[:, :, 2]
    mask_image[:, :, 1] = image[:, :, 1]
    mask_image[:, :, 2] = image[:, :, 0]

    for i in range(nCounts):
        color = (int(colors[i][0]*255.0 + 0.5), 
                 int(colors[i][1]*255.0 + 0.5), 
                 int(colors[i][2]*255.0 + 0.5))

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        
        y1, x1, y2, x2 = boxes[i]
        # draw box
        if show_bbox:
            cv2.rectangle(mask_image, (x1, y1), (x2, y2), color=color, thickness=1)
        
        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{}: {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        cv2.putText(mask_image, caption, (x1, y1+10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            mask_image = apply_mask(mask_image, mask, color)
        
        # Mask Polygon
        contour_mask = np.ones(mask.shape, dtype=np.uint8)
        contour_mask = mask_gray(contour_mask, mask)
        contours, hierarchy = cv2.findContours(contour_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
       # for i in range(len(contours)):
           #cv2.drawContours(mask_image, contours, i, color, 1)

    if save_image: 
        if output_image_path == None:
            output_image_path = '../images/cv2_result_images/result.jpg'
        cv2.imwrite(output_image_path, mask_image) 
    if show:
        cv2.imshow("result_image", mask_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return mask_image
