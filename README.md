# Handwritten Digit Recognition Project

## 1. Introduction

### Context and Objectives
Handwritten digit recognition is a key challenge in computer vision due to the variability in handwriting styles, quality, and the presence of noise or artifacts.

**Objective:**
This project aims to design a machine learning-based solution for recognizing handwritten digits. Key features of the application include:
- Drawing digits or uploading an image.
- Automatically segmenting detected digits.
- Training a supervised classification model.
- Predicting and displaying results interactively.

---

## 2. General Workflow of the Project

### Global Pipeline
1. A handwritten image is provided as input via the user interface.
2. The image undergoes preprocessing:
   - Grayscale conversion.
   - Color inversion.
   - Digit segmentation.
3. Segments are normalized, vectorized, and labeled for training or prediction.
4. Prediction results are displayed as classes for each segment.

**Expected Results:**
The model recognizes digits (0–9) and four additional calculation symbols with high accuracy, depending on the input quality and the trained model.

---

## 3. Data Preprocessing

### Initial Conversion
- **Grayscale Conversion:** Simplifies computations by removing color information.
- **Tools:** Pillow (PIL) and OpenCV are used for image manipulation.

### Segmentation
- The `segment_image` method splits the image into individual digit segments.
- Segments are resized to a fixed size of 50x50 pixels for uniformity.

### Normalization
- Pixel values are scaled between 0 and 1 by dividing by 255, accelerating model convergence during training.

---

## 4. Machine Learning Model

### Architecture
- **Input Layer:** 2,500 neurons (corresponding to 50x50 flattened pixels).
- **Hidden Layers:** One or more layers with ReLU (Rectified Linear Unit) activation functions.
- **Output Layer:** 14 neurons (10 digits + 4 calculation symbols).

### Functions Used
- **Activation Function:** ReLU introduces non-linearity in hidden layers.
- **Output Function:** Softmax converts predictions into probabilities.
- **Loss Function:** Cross-Entropy Loss quantifies the gap between predictions and true labels.

### Optimization
- The Adam optimizer is employed for its speed and robustness.

---

## 5. Model Training (Method: `train_model`)

### Training Data
- Images are organized in folders by class (e.g., `class0`, `class1`).
- Each image is converted to a 1D vector of size 2,500 and labeled.
- Approximately 100 examples per class are available, with potential for data augmentation.

### Training Process
- Input vectors and their labels are passed to the model via the `fit` method.
- The model adjusts its weights to learn characteristic patterns for each class.

---

## 6. Digit Prediction (Method: `predict_number`)

### Prediction Pipeline
1. A new image is temporarily saved and segmented into digits.
2. Each segment is:
   - Converted to a vector.
   - Normalized.
   - Passed to the model for prediction.

### Visualization
- Results are displayed segment by segment, along with associated probability scores.

---

## 7. Results

### Accuracy
- The model achieves ~91% accuracy on well-written digits.
- Performance decreases for poorly formed digits or segments containing multiple digits.

### Limitations
- The current ANN struggles with complex spatial relationships. Switching to a Convolutional Neural Network (CNN) could improve performance.

---

## 8. Potential Applications

### Administrative Automation
- Digitizing handwritten forms or documents.

### Education
- Interactive tools for learning handwriting and digit recognition.

### Accessibility
- Converting handwritten notes into digital text for users with writing difficulties.

---

## 9. Future Improvements

### Advanced Model
- Implement a CNN to exploit the spatial structure of images.

### Data Augmentation
- Enhance the dataset with transformations (e.g., rotation, stretching, noise).

### User Interface
- Add real-time visual feedback for predictions.
- Enable manual correction of segmentation errors.

---

## 10. Conclusion

**Summary:**
We developed a functional solution for handwritten digit recognition that integrates essential steps: segmentation, normalization, training, and prediction.

**Impact:**
This project showcases how AI can simplify complex tasks by combining computer vision and machine learning. With further improvements, it could find applications in education, administration, and accessibility.

---
https://youtu.be/sPcSD2fSxTo
