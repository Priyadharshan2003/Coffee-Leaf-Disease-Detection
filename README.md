# Coffee Leaf Disease Detection

This application uses deep learning to predict and classify coffee leaf diseases through image analysis. It helps farmers identify common coffee leaf diseases early, allowing for prompt intervention and improved crop management.

## Features

- Upload images of coffee leaves for disease detection
- Real-time prediction using a pre-trained CNN model
- Classification of common coffee leaf diseases:
  - Coffee Leaf Rust (CLR)
  - Coffee Berry Disease (CBD)
  - Cercospora Leaf Spot
  - Healthy leaves
- Visual feedback with prediction confidence scores
- Simple and intuitive user interface

## Technologies Used

- Python 3.8+
- Streamlit for the web interface
- TensorFlow/Keras for the deep learning model
- OpenCV for image preprocessing
- ResNet50 pre-trained model (fine-tuned for coffee leaf diseases)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/coffee-leaf-disease-detection.git
   cd coffee-leaf-disease-detection
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download the pre-trained model:
   - Download the model file from the releases page
   - Place it in the `models/` directory

## Usage

1. Start the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`

3. Upload an image of a coffee leaf through the interface

4. View the prediction results and disease information

## Project Structure

```
coffee-leaf-disease-detection/
├── app.py                   # Main Streamlit application
├── model_builder.py         # Script to build and train the model
├── utils.py                 # Utility functions for preprocessing
├── requirements.txt         # Project dependencies
├── models/                  # Directory for storing models
│   └── coffee_disease_model.h5
├── data/                    # Training and test data
│   ├── train/
│   └── test/
└── images/                  # Example images and UI assets
```

## Model Training

If you want to train your own model:

1. Prepare your dataset with labeled images in the `data/` directory
2. Run the training script:
   ```
   python model_builder.py --epochs 50 --batch_size 32
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Coffee Leaf Disease Dataset](https://www.kaggle.com/datasets/jorgebandeira/coffee-leaf-diseases-dataset)
- ResNet50 architecture by Microsoft Research
