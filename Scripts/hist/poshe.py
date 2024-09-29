import cv2
import numpy as np

def POSHE(image, block_size=(16, 16), overlap=0.5):
    h, w = image.shape
    step_x = int(block_size[0] * (1 - overlap))
    step_y = int(block_size[1] * (1 - overlap))
    output = np.zeros_like(image)
    
    for y in range(0, h, step_y):
        for x in range(0, w, step_x):
            x_end = min(x + block_size[0], w)
            y_end = min(y + block_size[1], h)
            block = image[y:y_end, x:x_end]
            block_equalized = cv2.equalizeHist(block)
            output[y:y_end, x:x_end] = block_equalized
    return output
