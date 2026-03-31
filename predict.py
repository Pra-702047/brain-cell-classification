import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

classes = ["Astrocyte","Neuron","NSC","Oligodendrocyte"]

model = load_model("models/brain_cell_classifier.h5")

def predict_image(img_path):

    img = image.load_img(img_path,target_size=(224,224))
    img = image.img_to_array(img)/255
    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img)

    result = classes[np.argmax(prediction)]

    print("Predicted Cell Type:",result)