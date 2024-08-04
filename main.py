import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2 as cv
from tensorflow.keras import datasets

# Load CIFAR-10 dataset for class names and test data
(_, _), (testing_images, testing_labels) = datasets.cifar10.load_data()
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
testing_images = testing_images / 255.0

# Load the trained CNN model
model = load_model('cnn_cifar10_model.h5')
print("CNN Model loaded successfully!")

# Evaluate the CNN model
test_loss, test_acc = model.evaluate(testing_images, testing_labels, verbose=2)
print(f"CNN Test accuracy: {test_acc:.2f}")

# Load and preprocess an external image for prediction
img = cv.imread('car-604019_640.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_resized = cv.resize(img, (32, 32))
img_array = np.expand_dims(img_resized, axis=0)  # Expand dims to match the shape of training data

# Make a prediction with CNN
predictions = model.predict(img_array)
index = np.argmax(predictions[0])
print(f"CNN Prediction is: {class_names[index]}")

# Display the image with CNN prediction
plt.imshow(img, cmap=plt.cm.binary)
plt.title(f"CNN Prediction: {class_names[index]}")
plt.show()
