import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import convolve2d
from PIL import Image
import argparse
import os

class ComputerVisionProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def load_image(self, image_path):
        """Load and preprocess image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image using PIL
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Convert to PyTorch tensor
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        return image_tensor, image_array
    
    def edge_detection(self, image_tensor):
        """Perform edge detection using Sobel operators"""
        print("Performing edge detection...")
        
        # Sobel operators
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(self.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).to(self.device)
        
        # Apply Sobel operators
        grad_x = F.conv2d(image_tensor, sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
        grad_y = F.conv2d(image_tensor, sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
        
        # Compute gradient magnitude
        edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold edges
        threshold = 0.1
        edges = (edge_magnitude > threshold).float()
        
        return edges.squeeze().cpu().numpy()
    
    def corner_detection(self, image_tensor):
        """Perform corner detection using Harris corner detector"""
        print("Performing corner detection...")
        
        # Convert to numpy for OpenCV processing
        image_np = image_tensor.squeeze().cpu().numpy()
        
        # Harris corner detection
        corners = cv2.cornerHarris(image_np, blockSize=2, ksize=3, k=0.04)
        
        # Threshold corners
        threshold = 0.01 * corners.max()
        corner_coords = np.where(corners > threshold)
        
        return corner_coords, corners
    
    def detect_keypoints(self, image_tensor):
        """Detect keypoints using SIFT-like approach"""
        print("Detecting keypoints...")
        
        # Convert to numpy for OpenCV processing
        image_np = image_tensor.squeeze().cpu().numpy()
        # print(image_np.shape)

        # Convert to uint8 format (0-255) as required by SIFT
        image_np = (image_np * 255).astype(np.uint8)
        
        ## Use SIFT detector
        sift = cv2.SIFT_create()
        keypoints = sift.detect(image_np, None)
        
        # Extract keypoint coordinates
        keypoint_coords = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.int32)
        
        return keypoints, keypoint_coords
    
    def create_gaussian_kernel(self, sigma, size):
        ##create 1D Gaussian kernel
        x = np.arange(-size//2, size//2 + 1)
        kernel = np.exp(-(x**2) / (2 * sigma**2))
        return kernel / kernel.sum()
    
    def create_dog_kernel(self, sigma1, sigma2, size):
        ##Create Difference of Gaussians kernel - gives a LoG
        g1 = self.create_gaussian_kernel(sigma1, size)
        g2 = self.create_gaussian_kernel(sigma2, size)
        return g1 - g2
    
    def laplacian_of_gaussian_2d(self, image_tensor, sigma=1.0):
        print("Computing Laplacian of Gaussian...")
        
        image_np = image_tensor.squeeze().cpu().numpy()
        
        # Create 1D LoG kernels using DoG approximation
        size = int(6 * sigma)
        if size % 2 == 0:
            size += 1
            
        # Create DoG kernels for x and y directions
        dog_x = self.create_dog_kernel(sigma, sigma * 1.6, size)
        dog_y = self.create_dog_kernel(sigma, sigma * 1.6, size)
        
        # Apply 2D convolution using 1D kernels (separable convolution)
        # First convolve with x kernel
        temp = convolve2d(image_np, dog_x.reshape(1, -1), mode='same')
        # Then convolve with y kernel
        log_result = convolve2d(temp, dog_y.reshape(-1, 1), mode='same')
        
        return log_result
    
    def identify_keypoint_scales(self, image_tensor, keypoint_coords, sigma_range=np.arange(0.5, 3.0, 0.5)):
        """Identify scale of keypoints using LoG response"""
        print("Identifying keypoint scales...")
        
        scales = {}
        responses = {}
        
        for sigma in sigma_range:
            log_response = self.laplacian_of_gaussian_2d(image_tensor, sigma)
            
            for i, (x, y) in enumerate(keypoint_coords):
                if 0 <= x < log_response.shape[1] and 0 <= y < log_response.shape[0]:
                    response = abs(log_response[y, x])
                    
                    if i not in responses:
                        responses[i] = {}
                    
                    responses[i][sigma] = response
        
        # Find dominant scale for each keypoint
        for kp_idx in responses:
            if responses[kp_idx]:
                best_sigma = max(responses[kp_idx].keys(), key=lambda s: responses[kp_idx][s])
                scales[kp_idx] = best_sigma
        
        return scales, responses
    
    def compute_keypoint_orientation(self, image_tensor, keypoint_coords, scales):
        ###Using SIFT to determine the dominant orientation of the keypoint
        print("Computing keypoint orientations...")
        
        image_np = image_tensor.squeeze().cpu().numpy()
        orientations = {}
        
        for i, (x, y) in enumerate(keypoint_coords):
            if i not in scales:
                continue
                
            sigma = scales[i]
            window_size = int(8 * sigma)
            
            # Extract region around keypoint
            y1, y2 = max(0, y - window_size), min(image_np.shape[0], y + window_size)
            x1, x2 = max(0, x - window_size), min(image_np.shape[1], x + window_size)
            
            if y2 - y1 < 2 or x2 - x1 < 2:
                continue
                
            region = image_np[y1:y2, x1:x2]
            
            # Compute gradients
            grad_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
            
            # Compute gradient magnitude and orientation
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            orientation = np.arctan2(grad_y, grad_x) * 180 / np.pi
            
            # Create orientation histogram
            hist, bins = np.histogram(orientation, bins=36, range=(-180, 180), weights=magnitude)
            
            # Find peaks in histogram
            peaks = []
            for j in range(1, len(hist) - 1):
                if hist[j] > hist[j-1] and hist[j] > hist[j+1] and hist[j] > 0.1 * hist.max():
                    peaks.append((j, hist[j]))
            
            # Sort peaks by magnitude
            peaks.sort(key=lambda x: x[1], reverse=True)
            
            # Find dominant orientations
            dominant_orientations = []
            if peaks:
                max_peak = peaks[0][1]
                dominant_orientations.append(peaks[0][0])
                
                # Check for secondary peaks above 70% threshold
                for j, peak_magnitude in peaks[1:]:
                    if peak_magnitude > 0.7 * max_peak:
                        dominant_orientations.append(j)
            
            # Convert bin indices to angles
            angles = []
            for bin_idx in dominant_orientations:
                angle = bins[bin_idx] + (bins[1] - bins[0]) / 2
                angles.append(angle)
            
            orientations[i] = angles
        
        return orientations
    
    def visualize_results(self, image_array, edges, corner_coords, keypoint_coords, 
                         scales, orientations, output_path="output_result.png"):
        ###Highlighting keypoints and their orientations
        print("Visualizing results...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image
        axes[0, 0].imshow(image_array, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Edge detection
        axes[0, 1].imshow(edges, cmap='gray')
        axes[0, 1].set_title('Edge Detection')
        axes[0, 1].axis('off')
        
        # Corner detection
        axes[1, 0].imshow(image_array, cmap='gray')
        if len(corner_coords[0]) > 0:
            axes[1, 0].scatter(corner_coords[1], corner_coords[0], c='red', s=20, alpha=0.7)
        axes[1, 0].set_title('Corner Detection')
        axes[1, 0].axis('off')
        
        # Final keypoints with orientations
        axes[1, 1].imshow(image_array, cmap='gray')
        
        # Draw keypoints with colored squares
        colors = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan', 'orange', 'purple']
        for i, (x, y) in enumerate(keypoint_coords):
            if i in scales and i in orientations:
                # Draw colored square
                color = colors[i % len(colors)]
                size = int(3 * scales[i])
                
                rect = plt.Rectangle((x - size//2, y - size//2), size, size, 
                                   linewidth=2, edgecolor=color, facecolor='none')
                axes[1, 1].add_patch(rect)
                
                # Draw orientation lines
                for angle in orientations[i]:
                    # Convert angle to radians
                    angle_rad = np.radians(angle)
                    length = size
                    dx = length * np.cos(angle_rad)
                    dy = length * np.sin(angle_rad)
                    
                    axes[1, 1].arrow(x, y, dx, dy, head_width=2, head_length=2, 
                                    fc=color, ec=color, linewidth=1)
        
        axes[1, 1].set_title('Keypoints with Orientations')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Results saved to {output_path}")
    
    def process_image(self, image_path, output_path="output_result.png"):
        """Main processing pipeline"""
        print(f"Processing image: {image_path}")
        
        # Load image
        image_tensor, image_array = self.load_image(image_path)
        
        # Edge detection
        edges = self.edge_detection(image_tensor)
        
        # Corner detection
        corner_coords, corner_response = self.corner_detection(image_tensor)
        
        # Keypoint detection
        keypoints, keypoint_coords = self.detect_keypoints(image_tensor)
        
        # Identify scales
        scales, responses = self.identify_keypoint_scales(image_tensor, keypoint_coords)
        
        # Compute orientations
        orientations = self.compute_keypoint_orientation(image_tensor, keypoint_coords, scales)
        
        # Print summary
        print(f"\nProcessing Summary:")
        print(f"Edges detected: {np.sum(edges > 0)}")
        print(f"Corners detected: {len(corner_coords[0])}")
        print(f"Keypoints detected: {len(keypoint_coords)}")
        print(f"Keypoints with scales: {len(scales)}")
        print(f"Keypoints with orientations: {len(orientations)}")
        
        # Visualize results
        self.visualize_results(image_array, edges, corner_coords, keypoint_coords, 
                             scales, orientations, output_path)
        
        return {
            'edges': edges,
            'corners': corner_coords,
            'keypoints': keypoint_coords,
            'scales': scales,
            'orientations': orientations
        }

def main():
    parser = argparse.ArgumentParser(description='Computer Vision Image Processing')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('--output', '-o', default='output_result.png', 
                       help='Output image path (default: output_result.png)')
    
    args = parser.parse_args()
    
    # Create processor and process image
    processor = ComputerVisionProcessor()
    
    try:
        results = processor.process_image(args.image_path, args.output)
        print(f"\nProcessing completed successfully!")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 