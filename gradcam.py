import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("models/brain_cell_classifier.h5")

def get_gradcam(img_path):

    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)/255
    img_array = np.expand_dims(img_array, axis=0)

    # Simple gradient visualization (no model.output issue)
    with tf.GradientTape() as tape:
        tape.watch(img_array)
        preds = model(img_array)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, img_array)[0]

    heatmap = tf.reduce_mean(grads, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img

    cv2.imwrite("gradcam_output.png", superimposed_img)

    print("GradCAM saved as gradcam_output.png")