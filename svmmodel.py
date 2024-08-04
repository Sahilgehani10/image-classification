import numpy as np
from sklearn import metrics
import joblib
import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.keras import datasets

# Load CIFAR-10 dataset for class names
(_, _), (testing_images, testing_labels) = datasets.cifar10.load_data()
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Load the trained SVM model, scaler, and PCA
svm_classifier = joblib.load('svm_cifar10_model.joblib')
scaler = joblib.load('scaler.joblib')
pca = joblib.load('pca.joblib')
print("SVM Model loaded successfully!")

# Preprocess the test dataset
testing_images = testing_images / 255.0
nsamples, nx, ny, nz = testing_images.shape
testing_images_flat = testing_images.reshape((nsamples, nx * ny * nz))
testing_images_flat = scaler.transform(testing_images_flat)
testing_images_pca = pca.transform(testing_images_flat)

# Evaluate the SVM model
predicted_labels = svm_classifier.predict(testing_images_pca)
svm_accuracy = metrics.accuracy_score(testing_labels, predicted_labels)
print(f"SVM Test accuracy: {svm_accuracy:.2f}")

# Load and preprocess an external image for prediction
img = cv.imread('car-604019_640.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_resized = cv.resize(img, (32, 32))
img_flattened = img_resized.flatten().reshape(1, -1)
img_flattened = scaler.transform(img_flattened)
img_pca = pca.transform(img_flattened)

# Make a prediction with SVM
prediction = svm_classifier.predict(img_pca)
index = prediction[0]
print(f"SVM Prediction is: {class_names[index]}")

# Display the image with SVM prediction
plt.imshow(img, cmap=plt.cm.binary)
plt.title(f"SVM Prediction: {class_names[index]}")
plt.show()
