# Siamese Network for Face Verification

This project demonstrates the use of a Siamese Network to perform face verification. A Siamese Network is a neural network that learns to distinguish between pairs of images, commonly used for tasks like image similarity or facial recognition.

## Project Overview

The goal of this project is to create a deep learning model that can recognize whether two images belong to the same person. The network uses a contrastive loss function to learn how to differentiate positive (same person) and negative (different person) pairs of images. This project is implemented using TensorFlow and Keras.

## Dataset

This project assumes that you have a dataset of face images and corresponding labels, with images saved in `face_images` and labels in `face_labels`. Labels indicate which images belong to the same person, allowing the model to learn similarity. 

## Code Structure

### Key Functions

- **generate_image_pairs(images, labels)**: Generates pairs of images, with positive pairs (same person) and negative pairs (different people), and assigns labels accordingly.
- **contrastive_loss(y, preds, margin=1)**: Defines the contrastive loss function, which calculates the distance between image embeddings to determine similarity.
- **visualize(image_pairs, labels, n=5)**: Visualizes a few positive and negative image pairs for checking dataset preparation.
- **test_visualize(images, n=5)**: Displays test images that will be compared with a reference image.

### Model Architecture

The Siamese Network is built using convolutional layers to extract features from images. The **embedding model** creates a high-dimensional feature vector representation of each image. Pairs of images are then compared by computing the Euclidean distance between their embeddings. The full model is trained with a binary cross-entropy loss function.

### Training and Testing

- **Training**: The model is trained on pairs of images, using binary labels to learn similarity between pairs.
- **Testing**: During testing, we can compare an anchor image to multiple other images and use the model to determine which images are similar.

### Performance Metrics

After training, we evaluate the model’s performance using:
- **Accuracy**: Measures the overall correct classification rate.
- **Recall**: Indicates how well the model identifies positive pairs.
- **F1 Score**: Balances precision and recall, giving an overall measure of performance.

Metrics are calculated as follows:

```python
from sklearn.metrics import accuracy_score, recall_score, f1_score

# Get predictions
pred_scores = siamese_model.predict([images_dataset[:, 0, :], images_dataset[:, 1, :]])
predicted_labels = (pred_scores > 0.5).astype(int).flatten()

# Calculate metrics
accuracy = accuracy_score(labels_dataset, predicted_labels)
recall = recall_score(labels_dataset, predicted_labels)
f1 = f1_score(labels_dataset, predicted_labels)

print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
```

## Running the Code

1. **Prepare the Dataset**: Load your images and labels into `face_images` and `face_labels` respectively.
2. **Generate Pairs**: Use `generate_image_pairs` to create image pairs and corresponding labels for training.
3. **Visualize Sample Pairs**: Optionally, use `visualize` to inspect a few image pairs.
4. **Train the Model**: Train the model by running the main training block with `siamese_model.fit()`.
5. **Evaluate**: Run the metric calculations to assess model performance.

## Requirements

- Python 3.x
- TensorFlow and Keras
- NumPy
- Matplotlib (for visualization)
- Scikit-learn (for metrics)

Install these libraries using pip:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Project Summary

This project explores the use of a Siamese Network for image verification tasks. By using deep learning techniques, the model learns to distinguish between pairs of face images. This approach can be extended to other verification tasks and serves as a foundation for building real-world face recognition systems.
