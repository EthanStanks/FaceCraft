'''
Credits for Human Head Semantic Segmentation: https://github.com/wiktorlazarski/head-segmentation
wiktorlazarski Wiktor Łazarski
9527-csroad hamihamiha
Szuumii Jakub Szumski
'''

import torch
from PIL import Image
import head_segmentation.segmentation_pipeline as seg_pipeline
import cv2
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
segmentation_pipeline = seg_pipeline.HumanHeadSegmentationPipeline(device=device)

def segment_head(img):
    # segment head and make background transparent
    np_image = np.array(img)
    segmentation_map = segmentation_pipeline.predict(np_image)
    segmentation_map_rgb = cv2.cvtColor(segmentation_map, cv2.COLOR_GRAY2RGB)
    segmented_region = np_image * segmentation_map_rgb
    alpha_channel = segmentation_map * 255
    rgba_image = np.dstack((segmented_region, alpha_channel))

    # remove transparent space around head
    coords = np.argwhere(alpha_channel)
    x_min, y_min = coords.min(axis=0)[:2]
    x_max, y_max = coords.max(axis=0)[:2]
    cropped = rgba_image[x_min:x_max+1, y_min:y_max+1]
    cropped = Image.fromarray(cropped)
    return cropped