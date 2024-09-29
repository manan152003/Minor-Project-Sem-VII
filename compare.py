import cv2
import matplotlib.pyplot as plt

original_image_path = "F://College//Minor Project - Shilpa Pandey//Datasets//DB1_B//101_1.tif"  # Original image path
processed_image_paths = [
    "F://College//Minor Project - Shilpa Pandey//Output_Images//global_db1_b//101_1.tif",  # Global image path
    "F://College//Minor Project - Shilpa Pandey//Output_Images//local_db1_b//101_1.tif",    # Local image path
    "F://College//Minor Project - Shilpa Pandey//Output_Images//poshe_db1_b//101_1.tif",    # POSHE image path
    "F://College//Minor Project - Shilpa Pandey//Output_Images//wthe_db1_b//101_1.tif",     # WTHE image path
    "F://College//Minor Project - Shilpa Pandey//Output_Images//agcwd_db1_b//101_1.tif"     # AGCWD image path
]

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

titles = ['Global', 'Local', 'POSHE', 'WTHE', 'AGCWD']
for i, path in enumerate(processed_image_paths):
    processed_image = cv2.imread(path, 0)

    plt.subplot(6, 2, 3 + 2*i)
    plt.imshow(processed_image, cmap='gray')
    plt.axis('off')
    plt.title(titles[i])

    plt.subplot(6, 2, 4 + 2*i)
    plt.hist(processed_image.ravel(), bins=256, range=[0, 255], color='gray')
    plt.title(f'Histogram - {titles[i]}')
    plt.xlim([0, 255])

plt.tight_layout()
plt.show()
