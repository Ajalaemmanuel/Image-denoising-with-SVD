# Image-denoising-with-SVD
## Table of content
* Overview
* Methodology
* Experiment
* Results
* Conclusion
* References 



## Overview 
Noise is absence of information; Image noise is a fundamental challenge in digital imaging, it represents loss of information in visual data and as given by second law of thermodynamics "Entropy of an isolated system always increases over time." In digital images noise manifests as random variations in pixel values that deviate from true image signal. Once information is lost it cannot be truly recovered according to information theory, but an estimate of lost information can be made from machine learning models by leveraging mathematical tools like Singular Value Decomposition (SVD).
How SVD solves it: Singular Value Decomposition (SVD) provides a powerful mathematical approach to image denoising by decomposing an image matrix into three fundamental components that reveal underlying structure of image data. 

**Theoretical Foundations** :
Image noise introduces random variations that deviate from the true signal. SVD provides a mathematical mechanism to decompose an image matrix A ∈ ℝᵐˣⁿ into three fundamental components:
A = UΣVᵀ
Where:

U represents left singular vectors
Σ is a diagonal matrix of singular values
V represents right singular vectors

**Noise Separation Mechanism** :
Isolating information related to noise in image data is the fundamental idea behind SVD-based denoising. The hierarchical ordering of singular values based on their magnitude enables a methodical approach to information preservation. The smaller singular values contain noise-related components, whereas the top-k largest singular values contain the most important image information.

To selectively truncate singular values, the method uses a strategic thresholding approach. To preserve the image's main structural elements, noise-carrying smaller singular values must be eliminated or reduced. By doing this, SVD successfully separates random noise from important image information.

**Algorithmic Approach**:
A methodical three-step process is used in the SVD denoising process. First, Singular Value Decomposition is used to break down the noisy image matrix. Second, the singular values are subjected to a meticulously crafted thresholding technique. Lastly, the reduced set of components is used to reconstruct the denoised image.

**Thresholding Techniques**:
SVD-based image denoising uses three main thresholding techniques. Complete elimination of singular values below a preset threshold is known as "hard thresholding." Soft thresholding reduces smaller singular values gradually. Adaptive thresholding dynamically modifies the threshold according to the unique properties of the image's noise.

**Advantages of SVD Denoising**:
When it comes to image denoising, SVD has several benefits. Mathematically speaking, it minimizes mean squared error while maintaining crucial image structures and offers provable convergence properties. In terms of computation, it makes use of effective linear algebraic operations, provides the possibility of parallel implementation, and exhibits exceptional scalability across a range of image sizes.

**Challenges and Limitations** :
Despite its merits, SVD-based image denoising is not without obstacles. With larger photos, the computational complexity rises dramatically. The best thresholding settings must be carefully chosen by researchers, necessitating advanced analytical techniques. A careful balance between noise reduction and information retention is also required because aggressive thresholding approaches may lead to the possible loss of tiny image details.

<pre> </pre>

## Methodology
SVD, known as Singular Value Decomposition is a type of data reduction tool that basically breaks down images into matrices. The steps use to achieve this includes decomposition, truncation, reconstruction, combining the channels (RGB)

**Decomposition**:
Let’s say we have an image A, it gets decomposed into 3 matrices as shown below: 
		A = U ∑ VT
These 3 matrices contain information about image A where U is an orthogonal matrix that represents pattern across rows of an image. It defines how the features vary vertically while VT also an orthogonal matrix that defines how the features vary horizontally and it captures the column space which is the horizontal structure of the image. ∑ is a diagonal matrix that contains the singular values and it defines the level of importance of each component of the image. The image below represents the first step if the image is in color;

 ![alt text](https://github.com/Ajalaemmanuel/Image-denoising-with-SVD/blob/main/Imagedenoising.png)



**Truncation**:
The diagonal matrix( ∑ ),  which contains the singular values tells us the most important parts of the image and we retain its top k singular values alongside its corresponding vectors in U and VT to reduce the size of the data and or denoise the image. The more singular value being kept, the closer it is to the original image. For example, if we keep 5 singular values, the image doesn’t look as clear as when we keep 20 singular values. How the select K also varies depending on the context, we look at the image complexity and the noise levels. In a case of high noise levels, it may benefit from lower K values. During this process,  an approximation Ak is derived as shown below 
Ak = Uk∑kVkT 
Uk represents the first column of U
∑k  represents the diagonal matrix of the largest singular  values
VkT represents the first row of V

**Reconstruction**:
In the reconstruction process, Uk ,  ∑k and VkT are combined to reconstruct Ak. These values are stored as opposed to storing the full matrix. 

![alt text](https://github.com/Ajalaemmanuel/Image-denoising-with-SVD/blob/main/Reconstruction.png)

The first column of U, largest values is ∑ and first row of V are picked in this stage. After this step, we evaluate the compression ratio and use  reconstruction error metrics like Peak Signal to Noise Ratio (PSNR), Mean Squared Error (MSE) to see how well the compression performed and also how clean the image is in a case of denoising. The formular used for the compression ratio is shown below; 

$$
\text{Compression Ratio} = \frac{\text{Original Image Size}} {\text{Compressed Image Size}}
$$


## Experiment 
**Setup**:
	The datasets used in this analysis were the Canadian Institute for Advanced Research, 10 classes (CIFAR-10) and the Modified National Institute of Standards and Technology database (MNIST). CIFAR-10 contains 60,000 32x32 images across 10 different classes. The dataset is broken into 50,000 training images and 10,000 testing images. MNIST is made up of 70,000 28x28 images of handwritten digits, with 60,000 training images and 10,000 testing images. The images in MNIST are all black and white images, while the CIFAR-10 images were converted to greyscale instead of remaining as their default color images. After researching different methods for downloading and unpacking each of these datasets, I decided to use the Python library TensorFlow (version 2.10.0) to load the datasets into my environment. The datasets can be easily loaded and split into training and testing sets through the tensorflow keras API with the function ‘load_data()’.

**Tools/Libraries**: 
Conda version 23.5.2 was used to install and maintain all required Python libraries. Libraries used in this analysis include: numpy (1.24.), matplotlib (3.5.3), scikit-image (0.20.0), scikit-learn (1.3.0), keras (2.10.0), tensorflow (2.10.0), and pandas (1.5.3). All programming for this assignment was done in a Jupyter notebook using Python version 3.8.18. 

**Metrics for Evaluation**: 
Several different methods were employed to thoroughly evaluate the performance of the image denoising process. Peak Signal-to-Noise Ratio (PSNR) measures the ratio between the power of a signal and the power of noise that affects the image quality. ## PSNR Formula

The Peak Signal-to-Noise Ratio (PSNR) is calculated using the following formula:

$$
\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right)
$$

Where:
- `MAX` is the maximum possible pixel value of the image (e.g., 255 for 8-bit images).
- `MSE` is the Mean Squared Error between the original and the distorted image.
  
Mean Squared Error (MSE) was also used as a metric to evaluate the performance of the model. MSE is useful for assessing the error between the original image and the reconstructed image by calculating the sum of squared differences. The formula for MSE is as follows: 

$$
\text{MSE} = \frac{1}{m \cdot n} \sum_{i=1}^{m} \sum_{j=1}^{n} [I(i, j) - I'(i, j)]^2
$$


Where:
- `m` and `n` are the dimensions of the image.
- `I(i, j)` is the pixel value of the original image at position (i, j).
- `I'(i, j)` is the pixel value of the compressed or noisy image at position (i, j).
  
The final method of performance evaluation used is the Structural Similarity Index (SSIM), which is used to measure the similarity between images on a scale ranging from -1 to 1. This method takes into account the luminance, contrast, and structure of an image. The formula for SSIM is: 

$$
\text{SSIM} = \frac{(2 \mu_I \mu_{I'} + C_1) (2 \sigma_{II'} + C_2)}{(\mu_I^2 + \mu_{I'}^2 + C_1) (\sigma_I^2 + \sigma_{I'}^2 + C_2)}
$$


Where:
- `μI` and `μI'` are the average pixel values of the original and compressed/noisy images.
- `σI` and `σI'` are the standard deviations of the pixel values of the original and compressed/noisy images.
- `σII'` is the covariance between the original and compressed/noisy images.
- `C1` and `C2` are constants to stabilize the division.

## Results 
**Performance Evaluation**:
The code performs SVD-based image denoising by testing different values of k (number of singular values retained). It evaluates the denoised images using three metrics:

* PSNR (Peak Signal-to-Noise Ratio): Measures image quality improvement compared to noisy images.
* MSE (Mean Squared Error): Quantifies the average error between the original and denoised images.
* SSIM (Structural Similarity Index): Assesses perceptual similarity between images.

In terms of lowering noise while maintaining image details, the SVD-based image denoising technique showed encouraging results. The reconstructed images showed better clarity and sharpness as the k value rose. Nonetheless, a trade-off between computational cost and denoising performance was noted.


![reconstructed image](https://github.com/user-attachments/assets/34d7f9a0-8c70-409c-83e2-fb7b2bd23cd3)

**Key Findings**:
Impact of k Value:

* Increasing the k value generally led to better denoising performance, as more significant singular values were retained.
* However, excessively large k values introduce overfitting and potential artifacts.
**Dataset Comparison** : 
The technique was effective on both CIFAR-10 and MNIST datasets, indicating its versatility.
The optimal k value might vary between datasets due to differences in image complexity and noise characteristics.

## In-Depth Performance Analysis:
**PSNR (Peak Signal-to-Noise Ratio)**: 
The PSNR calculates the ratio of a signal's maximum power to that of corrupting noise. Better quality is indicated by a higher PSNR. Despite its widespread use, PSNR has drawbacks. 
* It is quite sensitive to significant variations in pixels.
* Because it ignores structural information, it might not fairly represent perceived quality.
**MSE (Mean Squared Error)**:
The average squared difference between the values of the original and reconstructed pixels is determined by MSE. Better denoising performance is indicated by a lower MSE. Nevertheless, MSE has the **same drawbacks as PSNR**:
* It is susceptible to significant inaccuracies at the pixel level.
* It might not match up well with how people perceive the quality of images.
**SSIM (Structural Similarity Index)**:
A more complex statistic called SSIM considers structural data like structure, contrast, and brightness. It is more in line with how people perceive the quality of images. SSIM is a better option for evaluating the preservation of image information since it is less susceptible to small distortions and noise.
**Choosing the Right Metric**:
The application and the intended quality evaluation determine which measure is used. A more thorough assessment can frequently be obtained by combining many indicators. PSNR and SSIM together are frequently advised for SVD-based picture denoising:
* PSNR: Indicates the total amount of noise reduction.
* SSIM: Evaluates how well image structure and detail are maintained.
One can learn more about the efficacy of the denoising process by examining both measurements. For example, good noise reduction and detail preservation are indicated by high PSNR and SSIM. A low SSIM and a high PSNR, however, could indicate over-smoothing, in which detail is sacrificed in order to reduce noise.

**MNIST dataset**:
Images were taken from the MNIST dataset. To create a noisy environment, Gaussian noise was applied to the original photos. Three measures were used to assess the reconstructed images after the noisy images were subjected to the SVD-based denoising technique:

**Peak Signal-to-Noise Ratio (PSNR)**: Calculates the pixel-by-pixel difference between the original and reconstructed pictures. Better quality is indicated by a higher PSNR.
The average squared difference between the original and reconstructed pixels is determined by the Mean Squared Error (MSE). Better rebuilding is indicated by a lower MSE.
By taking into account elements including brightness, contrast, and structure, the Structural Similarity Index (SSIM) calculates how structurally similar the original and reconstructed images are. Better image detail preservation is indicated by a higher SSIM.

**Results**:
For a range of values of k, the number of singular values preserved during reconstruction, the graph shows the effectiveness of SVD-based denoising.

As k grows, PSNR and SSIM both significantly increase, suggesting that higher image quality results from keeping more singular values. But as k gets closer to greater values, the pace of improvement slows down. This implies that the key information in the image can be captured by a limited number of single values.
MSE: As k rises, the MSE gradually falls, suggesting that the reconstructed images get more accurate as the number of singular values increases.


![MNIST](https://github.com/user-attachments/assets/2ccde9f8-04e9-45b7-9cb3-3220b8331640)

**CIFAR dataset**:

Images were taken from the CIFAR-10 dataset. The SVD-based denoising technique was used after adding Gaussian noise to the original images. The performance was assessed using the same three metrics (PSNR, MSE, and SSIM).

The graph shows how well SVD-based denoising performs on the CIFAR-10 dataset for various values of k.

The pattern of PSNR and SSIM is comparable to that of the MNIST dataset, exhibiting a sharp rise at first, followed by a more gradual improvement. Nevertheless, the overall performance is worse than MNIST, suggesting that color image denoising is more difficult.
MSE: Like the MNIST dataset, the MSE slowly declines as k rises. The slower rate of decline, however, indicates that more unique values are required to attain similar denoising performance.

![CIFAR](https://github.com/user-attachments/assets/950c167c-ef48-40ae-949a-43b76d7ef9fd)

## Conclusion

On the MNIST and CIFAR-10 datasets, the SVD-based image denoising method has demonstrated efficacy in lowering noise and maintaining picture details. The selection of the k parameter, which regulates the number of singular values preserved during reconstruction, affects the technique's performance.
A lower k value can greatly enhance image quality for both datasets. However, the marginal performance improvement decreases as k rises. This implies that computing complexity and noise reduction need to be carefully balanced.
Even though the SVD-based method works well, more investigation is required to maximize the k selection procedure. The effectiveness of this methodology could be further improved by adaptive techniques that can automatically identify the ideal k value for various images and noise levels.

## References
Mean square error. Mean Square Error - an overview | ScienceDirect Topics. (n.d.). https://www.sciencedirect.com/topics/engineering/mean-square-error 

NATIONAL INSTRUMENTS CORP. (n.d.). Peak signal-to-noise ratio as an image quality metric. NI. https://www.ni.com/en/shop/data-acquisition-and-control/add-ons-for-data-acquisition-and-control/what-is-vision-development-module/peak-signal-to-noise-ratio-as-an-image-quality-metric.html 
