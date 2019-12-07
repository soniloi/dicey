import sys

import numpy
import PIL
import tensorflow
import tensorflow_hub

CLASSES = ["d4", "d6", "d8", "d10", "d12", "d20"]
IMAGE_SIZE = 150

if __name__ == "__main__":
    model_filename = sys.argv[1]
    image_filename = sys.argv[2]

    model = tensorflow.keras.models.load_model(
        model_filename,
        custom_objects={"KerasLayer": tensorflow_hub.KerasLayer}
    )

    image = PIL.Image.open(image_filename).resize((IMAGE_SIZE, IMAGE_SIZE))
    image = numpy.array(image)/255.0
    image = image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)
    predictions = model.predict(image)[0]

    for i in range(len(CLASSES)):
        print("({0}: {1:.2f}) ".format(CLASSES[i], predictions[i]), end=""),

    highest_prediction = numpy.argmax(predictions, axis=-1)
    print("predicted class: {0}".format(CLASSES[highest_prediction]))
