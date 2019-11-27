import os
import sys

import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator

CLASSES = ["d4", "d6", "d8", "d10", "d12", "d20"]
BATCH_SIZE = 100
IMAGE_SIZE = 150
EPOCHS = 1

if __name__ == "__main__":
    dataset_base_dir = sys.argv[1]
    train_split = sys.argv[2]
    valid_split = sys.argv[3]
    print(dataset_base_dir)

    train_dir = os.path.join(dataset_base_dir, train_split)
    valid_dir = os.path.join(dataset_base_dir, valid_split)

    for cls in CLASSES:
        class_train_dir = os.path.join(train_dir, cls)
        num_train_images = len(os.listdir(class_train_dir))
        print("number of {0} training images: {1}".format(cls, num_train_images))

        class_valid_dir = os.path.join(valid_dir, cls)
        num_valid_images = len(os.listdir(class_valid_dir))
        print("number of {0} validation images: {1}".format(cls, num_valid_images))

    train_generator = ImageDataGenerator(rescale=1./255)
    train_data = train_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                     class_mode="sparse")

    validation_generator = ImageDataGenerator(rescale=1./255)
    validation_data = validation_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                               directory=valid_dir,
                                                               shuffle=False,
                                                               target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                               class_mode="sparse")

    kernel_size = (3, 3)
    num_classes = len(CLASSES)
    model = tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Conv2D(8, kernel_size, activation="relu", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        tensorflow.keras.layers.MaxPooling2D(2, 2),
        tensorflow.keras.layers.Conv2D(16, kernel_size, activation="relu"),
        tensorflow.keras.layers.MaxPooling2D(2, 2),
        tensorflow.keras.layers.Conv2D(32, kernel_size, activation="relu"),
        tensorflow.keras.layers.MaxPooling2D(2, 2),
        tensorflow.keras.layers.Flatten(),
        tensorflow.keras.layers.Dense(512, activation="relu"),
        tensorflow.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    history = model.fit_generator(
        train_data,
        epochs=EPOCHS,
        validation_data=validation_data,
    )

    accuracy = history.history["accuracy"]
    loss = history.history["loss"]
    val_accuracy = history.history["val_accuracy"]
    val_loss = history.history["val_loss"]

    print("accuracy: {0} loss: {1} val_accuracy: {2} val_loss: {3}".format(
        accuracy, loss, val_accuracy, val_loss
    ))
