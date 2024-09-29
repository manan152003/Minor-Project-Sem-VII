import numpy as np

def AGCWD(image, alpha=0.75):
    image_norm = image / 255.0
    hist, _ = np.histogram(image_norm, bins=256, range=(0, 1), density=True)
    pdf = hist / np.sum(hist)
    cdf = np.cumsum(pdf)
    
    weighted_dist = np.power(cdf, alpha)
    gamma_corrected = np.interp(image_norm.flatten(), np.linspace(0, 1, 256), weighted_dist)
    gamma_corrected = gamma_corrected.reshape(image_norm.shape)
    
    img_enhanced = np.clip(gamma_corrected * 255, 0, 255).astype(np.uint8)
    return img_enhanced
