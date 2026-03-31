from dataset_loader import load_dataset
from model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint
import os

DATASET_PATH = "dataset"

def train_model():

    train_data, val_data = load_dataset(DATASET_PATH)

    model = build_model()

    # create models folder if not exists
    os.makedirs("models", exist_ok=True)

    # checkpoint callback
    checkpoint = ModelCheckpoint(
        "models/model_epoch_{epoch}.h5",
        save_freq='epoch',
        verbose=1
    )

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=10,
        callbacks=[checkpoint]
    )

    # final model save
    model.save("models/brain_cell_classifier.h5")

    print("Final model saved successfully!")

if __name__ == "__main__":
    train_model()