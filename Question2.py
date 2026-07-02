import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_color_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def k_means_segmentation(image, k):
    pixel_values = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_pixels = centers[labels.flatten()]
    segmented_image = segmented_pixels.reshape(image.shape)

    return segmented_image

# 1
img_original = get_color_image('./brain/onion.jpg')

# 2
K =  12
img_segmentada = k_means_segmentation(img_original, K)
img_save = cv2.cvtColor(img_segmentada, cv2.COLOR_RGB2BGR)
cv2.imwrite('resultadosQ2/output_onion_segmentada.png', img_save)

# 3
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Imagem Original")
plt.imshow(img_original)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Segmentação K-means (K={K})")
plt.imshow(img_segmentada)
plt.axis('off')

plt.show()