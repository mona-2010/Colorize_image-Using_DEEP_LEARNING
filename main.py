import cv2 as cv
import numpy as np
import os

# Load pre-trained model
proto_file = "./models/colorization_deploy_v2.prototxt"
model_file = "./models/colorization_release_v2.caffemodel"
numpy_file = "pts_in_hull.npy"

# Read network
net = cv.dnn.readNetFromCaffe(proto_file, model_file)
pts_in_hull = np.load(numpy_file).transpose().reshape(2, 313, 1, 1)

# Load model layers
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

# Load input grayscale image
image_path = "image.jpg"  # Change this
image = cv.imread(image_path)
if image is None:
    print("Error: Image not found!")
    exit()

# Convert image to LAB color space
lab = cv.cvtColor(image, cv.COLOR_BGR2Lab)
L_channel = lab[:, :, 0]  # Extract L channel

# Resize and normalize
L_resized = cv.resize(L_channel, (224, 224)) - 50
net.setInput(cv.dnn.blobFromImage(L_resized))

# Predict A and B channels
ab_output = net.forward()[0, :, :, :].transpose((1, 2, 0))
ab_output_resized = cv.resize(ab_output, (image.shape[1], image.shape[0]))

# Merge channels
lab_colorized = np.concatenate((L_channel[:, :, np.newaxis], ab_output_resized), axis=2)
colorized = cv.cvtColor(lab_colorized, cv.COLOR_Lab2BGR)

# Save and display result
output_path = "result.png"
cv.imwrite(output_path, (colorized * 255).astype(np.uint8))
print("Colorized image saved at:", output_path)

# Display output
cv.imshow("Colorized Image", colorized)
cv.waitKey(0)
cv.destroyAllWindows()
