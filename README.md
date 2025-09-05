# Computer Vision Image Processing Program

A comprehensive computer vision program built with PyTorch that performs edge detection, corner detection, keypoint detection, scale identification, and orientation analysis using SIFT-like algorithms.

## Features

### 1. Edge Detection
- Uses Sobel operators implemented with PyTorch convolutions
- Computes gradient magnitude and applies thresholding
- Efficient GPU acceleration when available

### 2. Corner Detection
- Implements Harris corner detector using OpenCV
- Automatically thresholds detected corners
- Provides corner coordinates for visualization

### 3. Keypoint Detection
- Uses SIFT algorithm for robust keypoint detection
- Extracts keypoint coordinates and properties
- Integrates with scale and orientation analysis

### 4. Scale Identification
- Implements Laplacian of Gaussian (LoG) using Difference of Gaussians (DoG)
- Uses separable 1D convolution for computational efficiency
- Tests multiple sigma values to find optimal scale for each keypoint

### 5. Orientation Analysis
- Computes gradient-based orientation histograms
- Identifies dominant orientations using peak detection
- Supports multiple orientations per keypoint (above 70% threshold)
- Uses SIFT-like approach for robust orientation estimation

### 6. Visualization
- Creates comprehensive 2x2 subplot visualization
- Shows original image, edges, corners, and final keypoints
- Color-coded keypoints with orientation arrows
- High-resolution output saving

## Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.8+
- NumPy 1.24+
- Matplotlib 3.7+
- Pillow 10.0+
- SciPy 1.11+
- scikit-image 0.21+

## Installation

1. **Install dependencies in a virtual env**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage
```bash
python computervision.py path/to/your/cat.jpg
```

### Custom Output Path
```bash
python computervision.py path/to/your/cat.jpg --output my_results.png
```


## Algorithm Details

### Laplacian of Gaussian (LoG)
The program implements LoG using a computationally efficient approach:
1. Creates 1D Difference of Gaussian (DoG) kernels
2. Applies separable 2D convolution (x-direction first, then y-direction)
3. Tests multiple sigma values to find optimal scale
4. Uses DoG as an approximation to true LoG for efficiency

### SIFT Orientation Analysis
1. Extracts local region around each keypoint
2. Computes gradient magnitude and orientation
3. Creates weighted orientation histogram (36 bins, -180° to +180°)
4. Identifies peaks above threshold
5. Supports multiple dominant orientations per keypoint

### Edge Detection
1. Applies Sobel operators in x and y directions
2. Computes gradient magnitude: √(Gx² + Gy²)
3. Thresholds result to create binary edge map

## Output

The program generates:
1. **Console output**: Processing progress and summary statistics
2. **Visualization**: 2x2 subplot showing all processing stages
3. **Saved image**: High-resolution PNG file with all results
4. **Return data**: Dictionary containing all processing results

## Performance Features

- **GPU acceleration**: Automatically uses CUDA if available
- **Separable convolution**: Efficient 2D LoG computation
- **Optimized kernels**: Minimal computational overhead
- **Memory efficient**: Processes images in chunks when needed

## Example Output

```
Using device: cuda
Processing image: sample.jpg
Performing edge detection...
Performing corner detection...
Detecting keypoints...
Computing Laplacian of Gaussian...
Identifying keypoint scales...
Computing keypoint orientations...
Visualizing results...

Processing Summary:
Edges detected: 15420
Corners detected: 45
Keypoints detected: 127
Keypoints with scales: 127
Keypoints with orientations: 127

Processing completed successfully!
Results saved to: output_result.png
```




## License

This project is open source and available under the MIT License.