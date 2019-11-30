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

    model.summary()

    image = PIL.Image.open(image_filename).resize((IMAGE_SIZE, IMAGE_SIZE))
    image = numpy.array(image)/255.0
    image = image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)
    predictions = model.predict(image)[0]

    for i in range(len(CLASSES)):
        print("class: {0} prediction: {1}".format(CLASSES[i], predictions[i]))

    highest_prediction = numpy.argmax(predictions, axis=-1)
    print("highest prediction: {0} predicted class: {1}".format(highest_prediction, CLASSES[highest_prediction]))
