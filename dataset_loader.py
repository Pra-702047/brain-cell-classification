from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_dataset(data_dir):

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_data = datagen.flow_from_directory(
        data_dir,
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        data_dir,
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    return train_data, val_data