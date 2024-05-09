from lime import lime_image
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

# Initialize LIME explainer
explainer = lime_image.LimeImageExplainer()

from ultralytics import YOLO
import os
import numpy as np
from PIL import Image
# Load a model
model = YOLO('models/yolov8m-seg.pt')  # load an official model

# Predict with the model
results = model.predict('D:/Uni/Yolov8-experiment/data/gtFine/images/val/lindau', save=True, imgsz=2048,
                        conf=0.5)  # predict on an image

# Define a function to predict with your YOLO model
def yolo_segmentation_predict(images):
    # Forward pass through YOLO segmentation model
    # Return segmented images
    segmented_images = []
    for image in images:
        # Placeholder: Assuming YOLO model returns segmented images
        segmented_images.append(np.ones_like(image) * 255)  # Placeholder for segmented image
    return segmented_images


# Define a function to load images from file paths
def load_images_from_paths(image_paths):
    images = []
    for path in image_paths:
        image = Image.open(path)
        images.append(np.array(image))
    return images

path= "D:/Uni/Yolov8-experiment/data/gtFine"
# Load train, val, and test image paths from files
train_paths = [line.strip() for line in open(path+"/train.txt", "r")]
val_paths = [line.strip() for line in open(path+"/val.txt", "r")]
test_paths = [line.strip() for line in open(path+"/test.txt", "r")]

# Load train images
val_images = load_images_from_paths(val_paths)

# Explain the predictions using LIME for a sample train image
explanation = explainer.explain_instance(np.squeeze(np.array(val_images[0])), yolo_segmentation_predict, top_labels=5,
                                         hide_color=0, num_samples=1000)

# Visualize the explanation
from skimage.segmentation import mark_boundaries

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                            hide_rest=True)
sample_image_with_mask = mark_boundaries(temp / 2 + 0.5, mask)
plt.imshow(sample_image_with_mask)
plt.show()
