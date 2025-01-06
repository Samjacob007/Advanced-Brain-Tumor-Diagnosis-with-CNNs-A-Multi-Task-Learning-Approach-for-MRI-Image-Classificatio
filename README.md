# Advanced-Brain-Tumor-Diagnosis-with-CNNs-A-Multi-Task-Learning-Approach-for-MRI-Image-Classification


## Project Overview
This project focuses on brain tumor classification from MRI images using deep learning models. The models implemented include VGG16, ResNet50, and a custom-built CNN. Additionally, an ensemble model combines the strengths of all three networks to enhance classification performance.

## Dataset
The dataset comprises MRI images divided into four classes:
- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor

A total of 7,023 MRI images were sourced and preprocessed to ensure uniformity. 

### Dataset Sources:
- Figshare Brain Tumor MRI dataset
- SARTAJ Brain MRI dataset
- Br35H MRI dataset



## Project Structure
brain-tumor-classification/
│
├── data/
│   ├── Training/
│   ├── Testing/
│
├── models/
│   ├── vgg_model.h5
│   ├── resnet_model.h5
│   ├── custom_cnn_model.h5
│
├── notebooks/
│   └── brain_tumor_classification.ipynb
│
└── README.md

## Requirements
To run this project, you need the following dependencies:
- Python 3.x
- TensorFlow 2.x
- Keras
- OpenCV
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib
- Seaborn
- Google Colab (for GPU acceleration)

Install dependencies using:

## Model Training and Evaluation
### 1. Data Preprocessing
- Images were resized to 224x224.
- Labels were one-hot encoded.
- Training, validation, and testing datasets were split (80-20 split for validation).
- Data augmentation was applied to avoid overfitting.
 - normalizing pixel values between 0 and 1 
## EDA
- Understand the distribution of tumor classes.
- Visualize pixel intensity distribution.
- Analyze image aspect ratios.
### 2. Model Architectures
**VGG16:**
- Transfer learning applied.
- Last five convolutional layers were fine-tuned.
- Fully connected layers with dropout and L2 regularization.

**ResNet50:**
- Transfer learning applied.
- Last 20 layers fine-tuned.
- Batch normalization and dropout added to prevent overfitting.

**Custom CNN:**
- Three convolutional blocks with max pooling and batch normalization.
- Fully connected layers with dropout for regularization.

### 3. Ensemble Model
- A weighted ensemble of VGG16, ResNet50, and the custom CNN.
- Voting weights: 50% (VGG16), 30% (ResNet50), 20% (Custom CNN).

### 4. Grad-CAM Visualization
- Grad-CAM was implemented for all models to visualize model focus during classification.
- Class activation maps were generated to interpret model predictions.

## Results
| Model                    | Accuracy | Precision | Recall | F1-Score |
|--------------------------|----------|-----------|--------|----------|
| VGG16                    | 94.0%    | 94.0%     | 94.0%  | 94.0%    |
| ResNet50                 | 80.0%    | 79.0%     | 79.0%  | 79.0%    |
| Custom CNN               | 77.0%    | 80.0%     | 76.0%  | 76.0%    |
| Ensemble (Weighted)      | **94.0%**| **94.0%** | **93.0%**| **93.0%**|

## Confusion Matrices
- Confusion matrices were generated for each model to evaluate classification performance for each tumor type.

## How to Use
1. Clone the repository:
https://github.com/Samjacob007/Advanced-Brain-Tumor-Diagnosis-with-CNNs-A-Multi-Task-Learning-Approach-forMRI-Image-Classification.git
2. Navigate to the project directory:
cd Brain-Tumor-Diagnosis-with-CNNs-A-Multi-Task-Learning-Approach-forMRI-Image-Classification
3. Run the Jupyter notebook or execute the Python script to train the models and generate predictions.



## Future Work

- **Multi-task learning** for segmentation and classification.  
- **Expand dataset** beyond 2,000 images; apply augmentation and synthetic data generation.  
- **Experiment with EfficientNet and Vision Transformers (ViTs).**  
- **Integrate SHAP and LIME** for enhanced model interpretability.  
- **Collaborate with healthcare institutions** for clinical validation and deployment.  


## Acknowledgments
Special thanks to the authors of the datasets and the open-source community for providing the resources required for this project.

