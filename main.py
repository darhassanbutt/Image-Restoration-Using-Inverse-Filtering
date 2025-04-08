import numpy as np
import cv2
from scipy import signal
from matplotlib import pyplot as plt 

def degrade_image(image, blur_kernel, noise_level):
  blurred_image = signal.convolve2d(image, blur_kernel, mode='same', boundary='wrap')
  noise = np.random.normal(0, noise_level, image.shape)
  degraded_image = blurred_image + noise
  return degraded_image

def reduce_noise(image, method='wiener', kernel_size=3):
    if method == 'median':
        return cv2.medianBlur(image.astype(np.uint8), kernel_size)
    elif method == 'wiener':
        return signal.wiener(image, mysize=kernel_size)
    else:
        raise ValueError("Invalid noise reduction method.")

def inverse_filter(degraded_image, blur_kernel):
    
    G = np.fft.fft2(degraded_image)
    H = np.fft.fft2(blur_kernel, s=degraded_image.shape)

    # Design inverse filter with regularization
    F_hat = G / (H + 1e-6)  # Add a small constant (e.g., 1e-6)

    # Perform Inverse Fourier Transform
    restored_image = np.fft.ifft2(F_hat).real
    restored_image = np.clip(restored_image, 0, 255)

    return restored_image.astype(np.uint8)

# Load the image
image = cv2.imread("aerials/2.1.01.tiff", cv2.IMREAD_GRAYSCALE)  # Replace with your image file

# Define the blur kernel (example: Gaussian blur)
blur_kernel = np.array([[1, 2, 1],
                      [2, 4, 2],
                      [1, 2, 1]]) / 16  

# Degrade the image
degraded_image = degrade_image(image, blur_kernel, noise_level=20)
noise_image = reduce_noise(degraded_image)
filter_image = inverse_filter(noise_image,blur_kernel)

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(degraded_image.astype(np.uint8), cmap='gray')
plt.title('Degraded Image')

plt.subplot(2, 2, 3)
plt.imshow(noise_image, cmap='gray')
plt.title('Restored Image')

plt.subplot(2, 2, 4)
plt.imshow(filter_image, cmap='gray')
plt.title('Restored Image')


plt.tight_layout()
plt.show()