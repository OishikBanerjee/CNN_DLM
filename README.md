MNIST Handwritten Digit Recognizer using CNN
Project Overview
This project implements a handwritten digit recognition system using a Deep Convolutional Neural Network (CNN). The system is designed to classify handwritten digits from the MNIST dataset, making it suitable for applications like postal sorting, bank check processing, and automated form digitization.

Contributors
Amrit Agarwal (055004)

Oishik Banerjee (055028)

Group No: 23

Problem Statement
Handwritten digit recognition is essential for various real-world applications but is often hindered by variations in handwriting styles. This project aims to address these challenges by developing a CNN-based model capable of improving classification accuracy through deep learning techniques. The solution focuses on automating feature extraction and enhancing generalization across diverse handwriting styles.

Abstract
The project employs a CNN architecture to classify handwritten digits effectively. Key features include:

Convolutional layers for feature extraction.

Pooling layers to reduce spatial dimensions.

ReLU activation for non-linear pattern detection.

Dropout layers to prevent overfitting.

The implementation leverages TensorFlow and Keras for efficient development, achieving high classification accuracy and demonstrating potential for real-world deployment in document processing systems.

Project Structure
Importing Libraries

Preparing the Dataset

Model Building

Model Fitting

Model Analysis

Predicting Using Test Data

Data Analysis
Dataset Preparation
Dataset: MNIST Handwritten Digit Recognition dataset.

Preprocessing steps:

Normalization: Scaled pixel values from to.

Reshaping: Converted pixel arrays into (28, 28, 1) matrices.

Encoding: Labels were one-hot encoded.

Train-Test Split: Divided data into training and validation sets.

Visualization
A countplot confirmed a balanced distribution of digit classes.

No missing values were identified.

Model Building
Architecture
The CNN model is based on the LeNet-5 architecture:
Input
→
[
[
Conv2D
→
ReLU
]
×
2
→
MaxPool2D
→
Dropout
]
×
2
→
Flatten
→
Dense
→
Dropout
→
Output
Input→[[Conv2D→ReLU]×2→MaxPool2D→Dropout]×2→Flatten→Dense→Dropout→Output

Key Features
Data Augmentation: Techniques like rotation, cropping, and flipping were applied to improve generalization.

Optimization Strategy: Used RMSProp optimizer for stable convergence and ReduceLROnPlateau to dynamically adjust the learning rate.

Model Training
Environment
Training was conducted on Kaggle's GPU-enabled environment for faster computation.

Performance Monitoring
Loss and accuracy metrics were tracked across epochs.

The learning curve indicated successful training with minimal overfitting.

Model Analysis
Evaluation Metrics
A confusion matrix highlighted misclassifications and provided insights for improvement.

The model achieved high accuracy but showed minor errors in specific digit classes.

Predictions
The trained model was used to classify the test dataset, with predictions stored in a CSV file for submission. Results aligned well with validation accuracy.

Managerial Insights
Scalability & Adaptability: The model can be extended to recognize characters in multiple languages.

Cost-Effectiveness: Reduces manual transcription costs by automating digit recognition.

Automation Potential: Ideal for banking, postal services, and form digitization systems.

Error Handling: Misclassifications can be addressed through active learning and periodic retraining.

Infrastructure Considerations: Requires GPU resources; cloud-based AI services can optimize deployment costs.
