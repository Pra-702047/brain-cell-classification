import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("models/brain_cell_classifier.h5")

def visualize_prediction(img_path):

    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)/255
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    # confidence score
    confidence = np.max(prediction)

    # create heatmap manually
    heatmap = np.mean(img_array[0], axis=-1)

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))

    output = heatmap * 0.4 + original

    cv2.imwrite("visual_output.png", output)

    print("Saved as visual_output.png")
    print("Confidence:", confidence)