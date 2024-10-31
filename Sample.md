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
  
