import gameVariables as var
from PIL import Image as PILImage
import numpy as np

pil_img = PILImage.open('images/cities/region/common_regions.png').convert('RGB')
aimg = np.array(pil_img)
y_len, x_len, _ = np.shape(aimg)
person_pos = {}

for person in var.region_quest_mapper:
    hex_value = var.inverse_region_colors[person]
    r, g, b = (int(hex_value[i:i + 2], 16) for i in (0, 2, 4))
    
    # Create a boolean mask where True represents pixels matching the target RGB
    matching_mask = np.all(aimg == (r, g, b), axis=-1)
    
    # Find the indices of pixels that match the target RGB value
    matching_indices = np.argwhere(matching_mask)
    
    # Convert the indices to a list of tuples (x, y positions)
    #matching_positions = [tuple(pos) for pos in matching_indices]
    
    # Determine the bounds of the matching pixels
    if matching_indices.size > 0:
        max_y = y_len - np.min(matching_indices[:, 0]) # invert to match kivy img indexes
        min_y = y_len - np.max(matching_indices[:, 0])
        min_x = np.min(matching_indices[:, 1])
        max_x = np.max(matching_indices[:, 1])
    else:
        min_y, max_y, min_x, max_x = (None, None, None, None)
    person_pos[person] = [min_x, max_x, min_y, max_y]