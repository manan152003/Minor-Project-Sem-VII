import cv2
import os
import argparse
from hist.poshe import POSHE
from hist.wthe import WTHE
from hist.agcwd import AGCWD

def global_histogram_equalization(image):
    return cv2.equalizeHist(image)

def adaptive_histogram_equalization(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def process_histogram_images(input_dir, output_dir, technique):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    temp = 0
    for filename in os.listdir(input_dir):
        if filename.endswith('.tif'):
            img_path = os.path.join(input_dir, filename)
            image = cv2.imread(img_path, 0)  

            if image is None:
                print(f"Failed to load {filename}")
                continue

            if technique == 'global':
                processed_image = global_histogram_equalization(image)
            elif technique == 'local':
                processed_image = adaptive_histogram_equalization(image)
            elif technique == 'poshe':
                processed_image = POSHE(image)
            elif technique == 'wthe':
                processed_image = WTHE(image)
            elif technique == 'agcwd':
                processed_image = AGCWD(image)
            else:
                raise ValueError("duh.")
            
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, processed_image)
            print(f"Processed and saved: {filename} at {output_dir}")

        temp += 1
        if temp >= 10:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=str,)
    parser.add_argument('technique', type=str, choices=['global', 'local', 'poshe', 'wthe', 'agcwd'])

    args = parser.parse_args()

    process_histogram_images('F:\\College\\Minor Project - Shilpa Pandey\\Datasets\\DB1_B', args.output_dir, technique=args.technique)