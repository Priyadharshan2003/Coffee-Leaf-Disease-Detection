import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import os
from PIL import Image
import io
import matplotlib.pyplot as plt
from utils import preprocess_image, segment_leaf, apply_clahe, extract_features

# Set page configuration
st.set_page_config(
    page_title="Coffee Leaf Disease Detection",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths to dataset
DATASET_PATH = "data"
TRAIN_IMAGES_PATH = os.path.join(DATASET_PATH, "train", "images")
TRAIN_MASKS_PATH = os.path.join(DATASET_PATH, "train", "masks")
TEST_IMAGES_PATH = os.path.join(DATASET_PATH, "test", "images")
TEST_MASKS_PATH = os.path.join(DATASET_PATH, "test", "masks")
EXAMPLE_IMAGES_PATH = os.path.join(DATASET_PATH, "examples") if os.path.exists(os.path.join(DATASET_PATH, "examples")) else None

# Define class names
CLASS_NAMES = ["Coffee Leaf Rust", "Coffee Berry Disease", "Cercospora Leaf Spot", "Healthy"]

# Load the pre-trained model
@st.cache_resource
def load_disease_model():
    try:
        model = load_model('models/coffee_disease_model.h5')
        return model
    except:
        st.error("Model file not found. Please ensure the model is in the 'models' directory.")
        return None

# Function to make predictions
def predict_disease(img, model):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])
    return predicted_class, confidence, predictions[0]

# Function to display disease information
def display_disease_info(disease_name):
    disease_info = {
        "Coffee Leaf Rust": {
            "causative_agent": "Hemileia vastatrix (fungus)",
            "symptoms": "Orange-yellow powder on the underside of leaves; yellow spots on the upper surface",
            "effects": "Leaf drop, reduced photosynthesis, weakened plants, reduced yield",
            "management": "Fungicides with copper compounds, resistant varieties, proper spacing for air circulation"
        },
        "Coffee Berry Disease": {
            "causative_agent": "Colletotrichum kahawae (fungus)",
            "symptoms": "Dark, sunken lesions on berries; brown/black spots on leaves",
            "effects": "Berry rot, premature fruit drop, significant yield loss",
            "management": "Copper-based fungicides, proper pruning, resistant varieties"
        },
        "Cercospora Leaf Spot": {
            "causative_agent": "Cercospora coffeicola (fungus)",
            "symptoms": "Brown spots with yellow halos on leaves; light-colored centers in advanced stages",
            "effects": "Defoliation, reduced plant vigor, lower yield quality",
            "management": "Fungicides, balanced nutrition, adequate spacing"
        },
        "Healthy": {
            "characteristics": "Vibrant green color, no spots or discoloration",
            "maintenance": "Regular monitoring, balanced fertilization, proper irrigation"
        }
    }
    
    info = disease_info[disease_name]
    
    if disease_name != "Healthy":
        st.subheader(f"About {disease_name}")
        st.write(f"**Causative Agent:** {info['causative_agent']}")
        st.write(f"**Symptoms:** {info['symptoms']}")
        st.write(f"**Effects:** {info['effects']}")
        st.write(f"**Management:** {info['management']}")
    else:
        st.subheader("Healthy Leaf")
        st.write(f"**Characteristics:** {info['characteristics']}")
        st.write(f"**Maintenance:** {info['maintenance']}")

# Function to find corresponding mask for an image
def find_mask_for_image(image_path, mask_dir):
    image_name = os.path.basename(image_path)
    # Assuming mask has same name as image
    mask_path = os.path.join(mask_dir, image_name)
    if os.path.exists(mask_path):
        return mask_path
    return None

# Function to display image processing steps
def display_image_processing(img, mask=None):
    # Convert PIL image to numpy array
    img_array = np.array(img)
    
    # Apply segmentation
    segmented_img = segment_leaf(img_array)
    
    # Apply CLAHE enhancement
    enhanced_img = apply_clahe(img_array)
    
    # Extract features
    features = extract_features(img_array)
    
    # Display the processing steps
    st.subheader("Image Processing Steps")
    
    if mask is not None:
        # If mask is provided, show a 2x2 grid
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        
        with col1:
            st.image(img_array, caption="Original Image", use_column_width=True)
        
        with col2:
            st.image(mask, caption="Ground Truth Mask", use_column_width=True)
            
        with col3:
            st.image(segmented_img, caption="Automated Segmentation", use_column_width=True)
            
        with col4:
            st.image(enhanced_img, caption="Enhanced Image (CLAHE)", use_column_width=True)
    else:
        # Without mask, show a 1x3 grid
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(img_array, caption="Original Image", use_column_width=True)
            
        with col2:
            st.image(segmented_img, caption="Leaf Segmentation", use_column_width=True)
            
        with col3:
            st.image(enhanced_img, caption="Enhanced Image (CLAHE)", use_column_width=True)
    
    # Display histogram of the image
    st.subheader("Image Histogram")
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # RGB histograms
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img_array], [i], None, [256], [0, 256])
        ax.plot(hist, color=color, label=f'{color.upper()} Channel')
    
    ax.set_xlim([0, 256])
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    st.pyplot(fig)
    
    # Display feature information
    st.subheader("Extracted Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Mean Pixel Value:** {features['mean']:.2f}")
        st.write(f"**Standard Deviation:** {features['std']:.2f}")
    
    with col2:
        # Calculate some additional metrics
        if 'hist' in features:
            hist_array = np.array(features['hist'])
            st.write(f"**Histogram Peak:** {np.argmax(hist_array)}")
            st.write(f"**Histogram Entropy:** {-np.sum(hist_array/np.sum(hist_array) * np.log2(hist_array/np.sum(hist_array) + 1e-10)):.2f}")

# Function to show region of interest analysis
def show_roi_analysis(img, mask=None):
    img_array = np.array(img)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Try to detect edges or regions of interest
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on original image
    contour_img = img_array.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    
    # Highlight potential disease regions
    # For demonstration, let's use a simple thresholding approach
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Define multiple color ranges for different diseases
    # Rust (orangish-yellow)
    lower_rust = np.array([15, 100, 100])
    upper_rust = np.array([30, 255, 255])
    rust_mask = cv2.inRange(hsv, lower_rust, upper_rust)
    
    # Berry disease (dark brown spots)
    lower_berry = np.array([0, 50, 20])
    upper_berry = np.array([10, 255, 100])
    berry_mask = cv2.inRange(hsv, lower_berry, upper_berry)
    
    # Cercospora (brown with yellow halo)
    lower_cercospora = np.array([20, 100, 100])
    upper_cercospora = np.array([40, 255, 255])
    cercospora_mask = cv2.inRange(hsv, lower_cercospora, upper_cercospora)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(rust_mask, berry_mask)
    combined_mask = cv2.bitwise_or(combined_mask, cercospora_mask)
    
    # Apply mask to image
    highlighted = img_array.copy()
    highlighted[combined_mask > 0] = [255, 0, 0]  # Highlight potential disease areas in red
    
    # Display the images
    st.subheader("Region of Interest Analysis")
    
    if mask is not None:
        # If mask is provided, show a comparison
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(contour_img, caption="Detected Edges & Contours", use_column_width=True)
        
        with col2:
            st.image(highlighted, caption="Potential Disease Regions", use_column_width=True)
            
        with col3:
            st.image(mask, caption="Ground Truth Mask", use_column_width=True)
            
        # Compare predicted vs ground truth
        mask_array = np.array(mask)
        if len(mask_array.shape) == 3:  # If mask is RGB
            mask_array = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
            mask_array = (mask_array > 0).astype(np.uint8) * 255
            
        # Calculate IoU or other metrics here if mask_array is a binary mask
        # For now, just show overlap
        overlap = cv2.bitwise_and(combined_mask, mask_array)
        overlap_img = img_array.copy()
        overlap_img[overlap > 0] = [0, 255, 0]  # Green for overlap
        
        st.image(overlap_img, caption="Prediction vs Ground Truth Overlap", use_column_width=True)
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(contour_img, caption="Detected Edges & Contours", use_column_width=True)
        
        with col2:
            st.image(highlighted, caption="Potential Disease Regions", use_column_width=True)
    
    # Count potential disease spots
    num_contours = len(contours)
    disease_area_percentage = (np.sum(combined_mask > 0) / (combined_mask.shape[0] * combined_mask.shape[1])) * 100
    
    st.write(f"**Number of detected regions:** {num_contours}")
    st.write(f"**Potentially affected area:** {disease_area_percentage:.2f}%")

# Function to get sample images from the dataset
def get_sample_images(num_samples=4):
    samples = []
    
    # Check if TEST_IMAGES_PATH exists and has images
    if os.path.exists(TEST_IMAGES_PATH):
        image_files = [f for f in os.listdir(TEST_IMAGES_PATH) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_files) >= num_samples:
            selected_files = np.random.choice(image_files, num_samples, replace=False)
            
            for file in selected_files:
                image_path = os.path.join(TEST_IMAGES_PATH, file)
                mask_path = find_mask_for_image(image_path, TEST_MASKS_PATH)
                
                # Try to determine class from filename or folder structure
                class_name = "Unknown"
                for class_idx, name in enumerate(CLASS_NAMES):
                    if name.lower().replace(" ", "_") in file.lower():
                        class_name = name
                        break
                
                samples.append((class_name, image_path, mask_path))
    
    # If we couldn't get samples from test directory, use example images
    if not samples and EXAMPLE_IMAGES_PATH and os.path.exists(EXAMPLE_IMAGES_PATH):
        image_files = [f for f in os.listdir(EXAMPLE_IMAGES_PATH) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for file in image_files[:num_samples]:
            image_path = os.path.join(EXAMPLE_IMAGES_PATH, file)
            
            # Try to determine class from filename
            class_name = "Unknown"
            for name in CLASS_NAMES:
                if name.lower().replace(" ", "_") in file.lower():
                    class_name = name
                    break
            
            samples.append((class_name, image_path, None))
    
    return samples

# Main application
def main():
    st.title("Coffee Leaf Disease Detection")
    
    # Sidebar with tabs
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Select Mode", ["Disease Detection", "Image Processing", "Dataset Explorer", "About"])
    
    # Sidebar information
    st.sidebar.title("About")
    st.sidebar.info(
        "This application uses deep learning to identify diseases in coffee leaves. "
        "Upload an image of a coffee leaf to get a prediction."
    )
    
    # Display dataset paths
    st.sidebar.title("Dataset Paths")
    st.sidebar.code(f"Training images: {TRAIN_IMAGES_PATH}\n"
                    f"Training masks: {TRAIN_MASKS_PATH}\n"
                    f"Testing images: {TEST_IMAGES_PATH}\n"
                    f"Testing masks: {TEST_MASKS_PATH}")
    
    # Model loading
    model = load_disease_model()
    
    if model is None:
        st.warning("Please make sure the model file is available before proceeding.")
        return
    
    # Disease Detection Mode
    if app_mode == "Disease Detection":
        st.header("Coffee Leaf Disease Detection")
        # File uploader
        uploaded_file = st.file_uploader("Upload an image of a coffee leaf", type=["jpg", "jpeg", "png"])
        
        col1, col2 = st.columns(2)
        
        if uploaded_file is not None:
            with col1:
                # Display the uploaded image
                image_data = uploaded_file.read()
                img = Image.open(io.BytesIO(image_data))
                st.image(img, caption="Uploaded Image", use_column_width=True)
                
                # Make prediction when user clicks the button
                if st.button("Predict Disease"):
                    with st.spinner("Analyzing image..."):
                        # Get prediction
                        predicted_class, confidence, all_predictions = predict_disease(img, model)
                        
                        # Display results
                        st.success(f"Prediction: **{CLASS_NAMES[predicted_class]}**")
                        st.info(f"Confidence: {confidence:.2%}")
                        
                        # Display all prediction probabilities
                        fig, ax = plt.subplots(figsize=(10, 5))
                        bars = ax.bar(CLASS_NAMES, all_predictions, color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'])
                        ax.set_ylabel('Probability')
                        ax.set_title('Prediction Confidence')
                        ax.set_ylim(0, 1)
                        
                        # Add values on top of bars
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                    f'{height:.2%}', ha='center', va='bottom')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
            
            with col2:
                # Display disease information after prediction
                if 'predicted_class' in locals():
                    display_disease_info(CLASS_NAMES[predicted_class])
        else:
            # Display example images from dataset
            st.write("### Sample Images from Dataset")
            samples = get_sample_images(4)
            
            if samples:
                # Create a 2x2 grid for example images
                for i in range(0, len(samples), 2):
                    cols = st.columns(2)
                    for j in range(2):
                        if i + j < len(samples):
                            class_name, img_path, _ = samples[i + j]
                            try:
                                cols[j].image(img_path, caption=class_name, use_column_width=True)
                            except:
                                cols[j].write(f"Sample image not found.")
            else:
                st.warning("No sample images found in the dataset. Please check the dataset paths.")
                
                # Use hardcoded examples as fallback
                st.write("### Example Images")
                example_images = [
                    ("Coffee Leaf Rust", "images/examples/rust.jpg"),
                    ("Coffee Berry Disease", "images/examples/berry_disease.jpg"),
                    ("Cercospora Leaf Spot", "images/examples/cercospora.jpg"),
                    ("Healthy", "images/examples/healthy.jpg")
                ]
                
                # Create a 2x2 grid for example images
                for i in range(0, len(example_images), 2):
                    cols = st.columns(2)
                    for j in range(2):
                        if i + j < len(example_images):
                            disease, img_path = example_images[i + j]
                            try:
                                cols[j].image(img_path, caption=disease, use_column_width=True)
                            except:
                                cols[j].write(f"Example image for {disease} not found.")
    
    # Image Processing Mode
    elif app_mode == "Image Processing":
        st.header("Image Processing Dashboard")
        st.write("This dashboard shows the steps involved in processing coffee leaf images for disease detection.")
        
        use_dataset_image = st.checkbox("Use image from dataset")
        
        if use_dataset_image:
            # Get sample images from the dataset
            samples = get_sample_images(8)
            
            if samples:
                options = [f"{class_name} ({os.path.basename(img_path)})" for class_name, img_path, _ in samples]
                selected_option = st.selectbox("Select an image from the dataset:", options)
                selected_idx = options.index(selected_option)
                
                _, selected_img_path, selected_mask_path = samples[selected_idx]
                
                img = Image.open(selected_img_path)
                mask = None
                if selected_mask_path and os.path.exists(selected_mask_path):
                    mask = Image.open(selected_mask_path)
            else:
                st.warning("No sample images found in the dataset.")
                img = None
                mask = None
        else:
            # File uploader
            uploaded_file = st.file_uploader("Upload an image for processing", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                # Display the uploaded image
                image_data = uploaded_file.read()
                img = Image.open(io.BytesIO(image_data))
                mask = None
            else:
                img = None
                mask = None
        
        if img is not None:
            # Display basic image information
            st.subheader("Image Information")
            st.write(f"**Image Size:** {img.width} x {img.height} pixels")
            st.write(f"**Image Format:** {img.format if hasattr(img, 'format') else 'Unknown'}")
            st.write(f"**Color Mode:** {img.mode}")
            
            # Create tabs for different processing views
            processing_tab, roi_tab, analysis_tab = st.tabs(["Processing Steps", "Region of Interest", "Analysis"])
            
            with processing_tab:
                display_image_processing(img, mask)
            
            with roi_tab:
                show_roi_analysis(img, mask)
            
            with analysis_tab:
                st.subheader("Disease Prediction")
                
                # Predict disease
                predicted_class, confidence, all_predictions = predict_disease(img, model)
                
                # Display results with gauge chart for confidence
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success(f"Prediction: **{CLASS_NAMES[predicted_class]}**")
                    st.info(f"Confidence: {confidence:.2%}")
                    
                    # Create gauge chart for confidence
                    fig = plt.figure(figsize=(8, 4))
                    ax = fig.add_subplot(111)
                    
                    # Draw gauge
                    ax.set_xlim(0, 10)
                    ax.set_ylim(0, 1)
                    ax.set_title("Confidence Level")
                    ax.set_yticks([])
                    ax.set_xticks([])
                    
                    # Create gauge background
                    for i in range(100):
                        ax.barh(0, 10, height=1, left=0, color='lightgray', alpha=0.3)
                    
                    # Create confidence bar
                    bar_length = confidence * 10
                    if confidence > 0.7:
                        color = 'green'
                    elif confidence > 0.4:
                        color = 'orange'
                    else:
                        color = 'red'
                    
                    ax.barh(0, bar_length, height=0.5, left=0, color=color)
                    
                    # Add labels
                    ax.text(0, -0.1, "0%", ha='center', va='center')
                    ax.text(5, -0.1, "50%", ha='center', va='center')
                    ax.text(10, -0.1, "100%", ha='center', va='center')
                    ax.text(bar_length, 0, f"{confidence:.1%}", ha='center', va='center', fontweight='bold')
                    
                    st.pyplot(fig)
                
                with col2:
                    # Display all prediction probabilities
                    fig, ax = plt.subplots(figsize=(8, 5))
                    bars = ax.bar(CLASS_NAMES, all_predictions, color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'])
                    ax.set_ylabel('Probability')
                    ax.set_title('Prediction Confidence by Disease Class')
                    ax.set_ylim(0, 1)
                    
                    # Add values on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{height:.2%}', ha='center', va='bottom')
                    
                    plt.xticks(rotation=15)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Display disease information
                display_disease_info(CLASS_NAMES[predicted_class])
    
    # Dataset Explorer Mode
    elif app_mode == "Dataset Explorer":
        st.header("Dataset Explorer")
        
        # Count images in dataset
        train_img_count = len([f for f in os.listdir(TRAIN_IMAGES_PATH) if f.endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(TRAIN_IMAGES_PATH) else 0
        train_mask_count = len([f for f in os.listdir(TRAIN_MASKS_PATH) if f.endswith(('.jpg', '.jpeg', '.png', '.tif'))]) if os.path.exists(TRAIN_MASKS_PATH) else 0
        test_img_count = len([f for f in os.listdir(TEST_IMAGES_PATH) if f.endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(TEST_IMAGES_PATH) else 0
        test_mask_count = len([f for f in os.listdir(TEST_MASKS_PATH) if f.endswith(('.jpg', '.jpeg', '.png', '.tif'))]) if os.path.exists(TEST_MASKS_PATH) else 0
        
        # Display dataset statistics
        st.subheader("Dataset Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Training Images", train_img_count)
            st.metric("Testing Images", test_img_count)
        
        with col2:
            st.metric("Training Masks", train_mask_count)
            st.metric("Testing Masks", test_mask_count)
        
        # Display class distribution if available
        try:
            class_counts = {"Unknown": 0}
            
            # Check training images
            if os.path.exists(TRAIN_IMAGES_PATH):
                for img_file in os.listdir(TRAIN_IMAGES_PATH):
                    if img_file.endswith(('.jpg', '.jpeg', '.png')):
                        found_class = False
                        for class_name in CLASS_NAMES:
                            if class_name.lower().replace(" ", "_") in img_file.lower():
                                if class_name not in class_counts:
                                    class_counts[class_name] = 0
                                class_counts[class_name] += 1
                                found_class = True
                                break
                        if not found_class:
                            class_counts["Unknown"] += 1
            
            # Plot class distribution
            if sum(class_counts.values()) > 0:
                st.subheader("Class Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                class_names = list(class_counts.keys())
                class_values = list(class_counts.values())
                
                bars = ax.bar(class_names, class_values, color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#CCCCCC'])
                
                # Add values on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{height}', ha='center', va='bottom')
                
                plt.xticks(rotation=15)
                plt.tight_layout()
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Error analyzing class distribution: {str(e)}")
        
        # Image browser
        st.subheader("Image Browser")
        
        dataset_section = st.radio("Select dataset section:", ["Training Images", "Test Images"])
        
        if dataset_section == "Training Images":
            browse_path = TRAIN_IMAGES_PATH
            mask_path = TRAIN_MASKS_PATH
        else:
            browse_path = TEST_IMAGES_PATH
            mask_path = TEST_MASKS_PATH
        
        if os.path.exists(browse_path):
            image_files = [f for f in os.listdir(browse_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            if image_files:
                # Display pagination controls
                items_per_page = 4
                total_pages = (len(image_files) + items_per_page - 1) // items_per_page
                
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    page_number = st.slider("Page", 1, max(1, total_pages), 1)
                
                start_idx = (page_number - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, len(image_files))
                
                # Display images for current page
                for i in range(start_idx, end_idx, 2):
                    cols = st.columns(2)
                    
                    for j in range(2):
                        if i + j < end_idx:
                            img_file = image_files[i + j]
                            img_path = os.path.join(browse_path, img_file)
                            
                            # Find corresponding mask
                            mask_file_path = find_mask_for_image(img_path, mask_path)
                            
                            # Determine class from filename if possible
                            img_class = "Unknown"
                            for class_name in CLASS_NAMES:
                                if class_name.lower().replace(" ", "_") in img_file.lower():
                                    img_class = class_name
                                    break
                            
                            cols[j].image(img_path, caption=f"{img_class}: {img_file}", use_column_width=True)
                            
                            # If mask exists, show button to view it
                            if mask_file_path and os.path.exists(mask_file_path):
                                if cols[j].button(f"View Mask for {img_file}", key=f"mask_{i+j}"):
                                    cols[j].image(mask_file_path, caption=f"Mask: {os.path.basename(mask_file_path)}", use_column_width=True)
            else:
                st.warning(f"No images found in {browse_path}")
        else:
            st.warning(f"Directory not found: {browse_path}")