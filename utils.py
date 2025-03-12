import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

def preprocess_image(img, target_size=(224, 224)):
    """
    Preprocess the image for model prediction
    
    Args:
        img: PIL Image object
        target_size: tuple of (height, width) for resizing
        
    Returns:
        Preprocessed image array ready for model input
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    # Convert to RGB if grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Resize image
    img = cv2.resize(img, target_size)
    
    # Normalize pixel values
    img = img / 255.0
    
    # Expand dimensions to match model input shape
    img = np.expand_dims(img, axis=0)
    
    return img

def segment_leaf(img):
    """
    Segment the leaf from the background
    
    Args:
        img: Input image as numpy array
        
    Returns:
        Segmented leaf image
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Define range of green color in HSV
    lower_green = np.array([20, 20, 20])
    upper_green = np.array([100, 255, 255])
    
    # Create mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Bitwise-AND mask and original image
    segmented = cv2.bitwise_and(img, img, mask=mask)
    
    return segmented

def apply_clahe(img):
    """
    Apply Contrast Limited Adaptive Histogram Equalization
    
    Args:
        img: Input image as numpy array
        
    Returns:
        Enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    # Split LAB channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels back
    lab = cv2.merge((l, a, b))
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return enhanced

def extract_features(img):
    """
    Extract features from the image for additional analysis
    
    Args:
        img: Input image as numpy array
        
    Returns:
        Dictionary of extracted features
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Calculate mean and standard deviation
    mean, std = cv2.meanStdDev(gray)
    
    # Calculate texture features using GLCM
    # Note: This would require additional libraries like scikit-image
    
    features = {
        'mean': float(mean[0][0]),
        'std': float(std[0][0]),
        'hist': hist.flatten().tolist()
    }
    
    return features

def data_augmentation(img):
    """
    Apply data augmentation to the image
    
    Args:
        img: Input image as numpy array
        
    Returns:
        List of augmented images
    """
    augmented = []
    
    # Original image
    augmented.append(img)
    
    # Flip horizontally
    augmented.append(cv2.flip(img, 1))
    
    # Rotate 90 degrees
    augmented.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    
    # Adjust brightness
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * 1.2  # Increase brightness by 20%
    augmented.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    
    return augmented
