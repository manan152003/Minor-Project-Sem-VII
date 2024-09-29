import cv2
import numpy as np

def WTHE(image, low_thresh=0.001, high_thresh=0.999, gamma=1.2):
    hist, bins = np.histogram(image.flatten(), 256, [0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    
    low_bound = np.searchsorted(cdf_normalized, low_thresh * cdf_normalized[-1])
    high_bound = np.searchsorted(cdf_normalized, high_thresh * cdf_normalized[-1])
    
    img_clipped = np.clip(image, low_bound, high_bound)
    img_gamma_corrected = np.power(img_clipped / np.max(img_clipped), gamma)
    img_equalized = cv2.equalizeHist((img_gamma_corrected * 255).astype(np.uint8))
    
    return img_equalized
