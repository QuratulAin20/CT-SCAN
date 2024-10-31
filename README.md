# De-noising CT-SCAN-IMAGE 
## Workflow if you are working with Big Data
## 1. Data Preparation
**1.1. Collect Data**
Images must have 25 sets organized in a structured format (e.g., folders for each set).

**1.2. Preprocess Data**
Normalize the images (e.g., scale pixel values to [0, 1] or [0, 255]).
Resize the images if necessary to ensure uniformity across your dataset.

**1.3. Visualize Data**
Display sample images from each set to getting insight.

## 2. Add Noise to Images
**2.1. Define Noise Parameters**
Choose the type of noise to add and define parameters such as mean and standard deviation (sigma). Type of Noise
Gaussian Noise: Simulates sensor noise commonly found in medical imaging.
Salt and Pepper Noise: Useful for testing the effectiveness of denoising algorithms against impulsive noise.
Poisson Noise: Simulates noise found in low-light imaging conditions.
There are also more type of noise but not relevant to this project
Example code
```python
def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image
```
**2.2. Generate Noisy Images**
Apply the noise to each set of CT scan images and store the noisy images for further processing.

## 3. Denoise Images Using Different Models
**3.1. Denoising Models**

*Models for denoising:*
CNN
DnCNN
U-Net
Variational Autoencoder (VAE)
BM3D (Block-Matching and 3D Filtering)
Non-Local Means
GAN-based models

**3.2. Train Denoising Models**
For deep learning models (CNN, DnCNN, U-Net):
Split the dataset into training, validation, and test sets.
Train the models using the noisy images as input and the original images as targets.
Save the trained models.

**3.3. Apply Denoising**
Use each model to denoise the noisy images and store the results.

## 4. Evaluate Denoised Images
**4.1. Define Evaluation Metrics**
Choose qualitative metrics (visual inspection) and quantitative metrics (e.g., PSNR, SSIM).
```python
def PSNR(original, denoised):
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return 100  # Perfect similarity
    max_pixel = 255.0
    return 20 * log10(max_pixel / sqrt(mse))

def SSIM(original, denoised):
    return structural_similarity(original, denoised, data_range=denoised.max() - denoised.min())
```

**4.2. Calculate Metrics**
For each denoised image, calculate PSNR and SSIM values compared to the original images.
```python
for i in range(num_images):
    psnr_value = PSNR(original_images[i], denoised_images[i])
    ssim_value = SSIM(original_images[i], denoised_images[i])
    print(f'Image {i+1} - PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}')
```

**4.3. Visualize Results**
Create a side-by-side comparison of original, noisy, and denoised images for qualitative evaluation.
```python
plt.subplot(1, 3, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')

plt.subplot(1, 3, 3)
plt.imshow(denoised_image, cmap='gray')
plt.title('Denoised Image')

plt.show()
```
## 5. Analyze Results
**5.1. Compare Models**
Analyze the PSNR and SSIM results to determine which model performed best for denoising.
CT-SCAN/Sample.md at main · QuratulAin20/CT-SCAN
