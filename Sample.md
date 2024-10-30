# Author: Quratul Ain
# TITILE: Denoising of CT-SCAN using Autoencoders

**Email: qurat-zee@hotmail.com**

## NOTEBOOK
**This notebook is only a sample to denoise the image of CT-SCAN dataset**
[CT-SCAN | Kaggle](https://www.kaggle.com/code/quratulain20/ct-scan?scriptVersionId=204211400)

# Results


#### Overview of PSNR Values

The PSNR (Peak Signal-to-Noise Ratio) values for the various denoising methods applied to the CT scan images are summarized as follows:

- **Autoencoder Image**: 60.90 dB
- **Median Filter Image**: 60.37 dB
- **Gaussian Filter Image**: 60.28 dB
- **Average Filter Image**: 60.54 dB
- **Bilateral Filter Image**: 60.82 dB
- **BM3D Filter Image**: 60.92 dB

#### Analysis of Results

The PSNR values indicate the effectiveness of each denoising method in reconstructing the original CT images from their noisy counterparts. Higher PSNR values suggest better image quality and less distortion.

1. **Performance of Autoencoder**:
   - The autoencoder achieved a PSNR of **60.90 dB**, making it one of the top performers in this evaluation. This indicates that the model effectively preserved the structural integrity of the CT scans while reducing noise.

2. **Comparison with Traditional Filters**:
   - Among traditional filtering techniques, the **Bilateral Filter** yielded a PSNR of **60.82 dB**, which is slightly lower than the autoencoder but still demonstrates strong performance in edge preservation and noise reduction.
   - The **Median Filter** and **Average Filter** had PSNR values of **60.37 dB** and **60.54 dB**, respectively. While both methods were effective, they did not perform as well as the autoencoder and bilateral filter, indicating that they may not preserve fine details as effectively.

3. **BM3D Filtering**:
   - The **BM3D Filter** achieved a PSNR of **60.92 dB**, the highest among all methods tested. This reinforces the effectiveness of BM3D in handling Gaussian noise and restoring image quality, leveraging its advanced block-matching and collaborative filtering techniques.

4. **Gaussian Filter Performance**:
   - The **Gaussian Filter** produced a PSNR of **60.28 dB**, which was the lowest among the evaluated methods. This suggests that while Gaussian filtering can smooth images, it may introduce blurring and loss of detail, particularly in medical imaging where precision is critical.

#### Conclusion

In conclusion, the results highlight that the traditional filters can be effective for basic noise reduction, advanced methods such as autoencoders and BM3D  outperform in terms of PSNR. The superior performance of these models demonstrates their capability to preserve important structural details in CT scan images while effectively reducing noise.


---

### Workflow 
---
#### 1. **Data Preparation**

- **1.1. Collect Data**
  - Images must have 25 sets organized in a structured format (e.g., folders for each set).

- **1.2. Preprocess Data**
  - Normalize the images (e.g., scale pixel values to [0, 1] or [0, 255]).
  - Resize the images if necessary to ensure uniformity across your dataset.

- **1.3. Visualize Data**
  - Display sample images from each set to getting insight.

#### 2. **Add Noise to Images**

- **2.1. Define Noise Parameters**
  - Choose the type of noise to add and define parameters such as mean and standard deviation (sigma).
  **Type of Noise**
  - Gaussian Noise: Simulates sensor noise commonly found in medical imaging.
  - Salt and Pepper Noise:  Useful for testing the effectiveness of denoising algorithms against impulsive noise.
  - Poisson Noise: Simulates noise found in low-light imaging conditions.\
  *There are also more type of noise but not relevant to this project* 
---
  > Example code
---
```python
def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image
```

- **2.3. Generate Noisy Images**
  - Apply the noise to each set of CT scan images and store the noisy images for further processing.

#### 3. **Denoise Images Using Different Models**

- **3.1. Denoising Models**
  - Models for denoising:
    - CNN
    - DnCNN
    - U-Net
    - Variational Autoencoder (VAE)
    - BM3D (Block-Matching and 3D Filtering)
    - Non-Local Means
    - GAN-based models

- **3.2. Train Denoising Models**
  - For deep learning models (CNN, DnCNN, U-Net):
    - Split the dataset into training, validation, and test sets.
    - Train the models using the noisy images as input and the original images as targets.
    - Save the trained models.

- **3.3. Apply Denoising**
  - Use each model to denoise the noisy images and store the results.

#### 4. **Evaluate Denoised Images**

- **4.1. Define Evaluation Metrics**
  - Choose qualitative metrics (visual inspection) and quantitative metrics (e.g., PSNR, SSIM).

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

- **4.2. Calculate Metrics**
  - For each denoised image, calculate PSNR and SSIM values compared to the original images.

```python
for i in range(num_images):
    psnr_value = PSNR(original_images[i], denoised_images[i])
    ssim_value = SSIM(original_images[i], denoised_images[i])
    print(f'Image {i+1} - PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}')
```

- **4.3. Visualize Results**
  - Create a side-by-side comparison of original, noisy, and denoised images for qualitative evaluation.

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

#### 5. **Analyze Results**

- **5.1. Compare Models**
  - Analyze the PSNR and SSIM results to determine which model performed best for denoising.
  