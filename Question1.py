import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_gray_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img

def gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def median_filter(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)

def histogram_plot_and_save(image, output_path='histograma.png'):
    plt.figure()
    plt.hist(image.ravel(), bins=256, range=(0, 256), color='gray')
    plt.title('Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(output_path)
    plt.show()

def limiarizacao_otsu(image):
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

def opening(image, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def closing(image, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def isolar_tumor(binary_image):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image)
    componentes = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    componentes_ordenados = sorted(componentes, key=lambda x: x[1], reverse=True)
    tumor_idx, tumor_area = componentes_ordenados[1]
    tumor_isolado = np.zeros_like(binary_image)
    tumor_isolado[labels == tumor_idx] = 255
    return tumor_isolado, tumor_area


# 1 e 2
img_original = get_gray_image('./brain/brain.jpg')
    
# 3
img_suave = gaussian_blur(img_original, 5)
img_suave = median_filter(img_suave, 5)

# 4
histogram_plot_and_save(img_suave, 'resultadosQ1/output_histograma.png')
img_binaria = limiarizacao_otsu(img_suave)
cv2.imwrite('resultadosQ1/output_binaria.png', img_binaria)

# 5
img_morf = opening(img_binaria, 5)
img_morf = closing(img_morf, 5)
cv2.imwrite('resultadosQ1/output_morfologia.png', img_morf)

# 6
tumor_mask, area = isolar_tumor(img_morf)
cv2.imwrite('resultadosQ1/output_tumor_isolado.png', tumor_mask)
resultado_sobreposto = cv2.bitwise_and(img_original, img_original, mask=tumor_mask)
cv2.imwrite('resultadosQ1/output_resultado_final.png', resultado_sobreposto)


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Imagem Original")
plt.imshow(img_original, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Tumor Isolado")
plt.imshow(resultado_sobreposto, cmap='gray')
plt.axis('off')

plt.show()