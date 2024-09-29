import cv2
import os
import matplotlib.pyplot as plt

original_img_dir = "F://College//Minor Project - Shilpa Pandey//Datasets//DB1_B"
output_dir = {
    'Global': "F://College//Minor Project - Shilpa Pandey//Output_Images//global_db1_b",
    'Local': "F://College//Minor Project - Shilpa Pandey//Output_Images//local_db1_b",
    'POSHE': "F://College//Minor Project - Shilpa Pandey//Output_Images//poshe_db1_b",
    'WTHE': "F://College//Minor Project - Shilpa Pandey//Output_Images//wthe_db1_b",
    'AGCWD': "F://College//Minor Project - Shilpa Pandey//Output_Images//agcwd_db1_b"
}
original_images = [f for f in os.listdir(original_img_dir) if f.endswith('.tif')][:10] 

for idx, original_image_name in enumerate(original_images):
    original_image_path = os.path.join(original_img_dir, original_image_name)
    original_image = cv2.imread(original_image_path, 0)

    plt.figure(figsize=(15, 12))

    plt.subplot(6, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')
    plt.title('Original Image')

    plt.subplot(6, 2, 2)
    plt.hist(original_image.ravel(), bins=256, range=[0, 255], color='gray')
    plt.title('Histogram - Original')
    plt.xlim([0, 255])

    for i, (title, folder) in enumerate(output_dir.items()):
        processed_image_path = os.path.join(folder, original_image_name)
        processed_image = cv2.imread(processed_image_path, 0)

        plt.subplot(6, 2, 3 + 2 * i)
        plt.imshow(processed_image, cmap='gray')
        plt.axis('off')
        plt.title(title)

        plt.subplot(6, 2, 4 + 2 * i)
        plt.hist(processed_image.ravel(), bins=256, range=[0, 255], color='gray')
        plt.title(f'Histogram - {title}')
        plt.xlim([0, 255])

    plt.tight_layout()
    save_path = f"F://College//Minor Project - Shilpa Pandey//plots//{original_image_name}_plot.png" 
    plt.savefig(save_path)
    plt.close()  