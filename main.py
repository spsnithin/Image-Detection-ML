import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the model
module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/5"
model = hub.load(module_handle)

# Load labels
labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
)
with open(labels_path, 'r') as f:
    imagenet_labels = f.read().splitlines()

# Predict
def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = img_array[np.newaxis, ...]
    
    prediction = model(tf.constant(img_array, dtype=tf.float32))
    predicted_index = np.argmax(prediction)
    return imagenet_labels[predicted_index]

# Run
if __name__ == "__main__":
    image_path = input("Enter path to image (e.g., charminar.jpg): ")
    label = predict_image(image_path)
    print("Predicted Label:", label)

    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(label)
    plt.axis('off')
    plt.show()
