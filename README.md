# dicey

This is a basic experiment in using machine learning to identify images of dice. It is written in python, and uses the TensorFlow library.

# First phase

This will involve identifying the shape of a die, with the following restrictions:

* There will be exactly one die in the image (not zero and not multiples).
* The die will be one of a common RPG set, i.e. d4, d6, d8, d10, d12, d20.
* The symbols on the faces will be numerals (not pips or other symbols).

# Dataset

I created my own dataset for this. This involved varying the following.

* Assorted types of dice (solid, mottled, etc.) of each of the six categories.
* Background surfaces (solid white, solid black, wood, glass).
* Lighting (natural, artificial, mixed).

To achieve this, I took short videos to vary the perspective on each die, then split the videos into images. The images were then cropped to be the same square in size, and turned to greyscale.

In all, I generated over 9,000 images. I split these into a training set and a validation set, in a ratio of 80/20. The dataset was not significantly weighted in favour of any category.

# Procedure

This involved creating a simple CNN. It consists of three sets of convolution/maxpooling, followed by a flattening, followed by a dense layer.

Overfitting was identified quite early. At one point, validation accuracy was lagging training accuracy by about 10%. In order to combat it, some data augmentation is performed, varying the rotation, zoom, etc. of training images. In addition, the model features a dropout layer.

# Results

## Training and validation

This process was able to achieve good training and validation accuracy. The validation accuracy did lag the training accuracy consistently by a small amount though, suggesting that some level of overfitting persisted.

Note that, while the validation images were not the same as the training images, they would have been generated from the same original videos.

## Manual testing

For the purposes of manual testing, I created a series of stand-alone images under slightly different conditions from the original training and validation sets. I then ran these against a model saved from the training phase.

The results were less impressive at this point (only a little over 40% at its best). However, there were a few observations to make from them.

* d4 was by far the most consistently correctly guessed category. d4 dice were mostly predicted correctly, and dice of other categories were seldom identified as d4.
* d6 dice were mostly correctly identified, but many other categories were also identified as d6.
* d8, d12, and d20 had some measure of success. They were consistently predicted more often than 1/6th of the time, but usually not over 50% of the time.
* d10 was by far the most problematic. d10 dice were pretty much never identified as such (not even 1/6th of the time that one might expect from a blind guess). Nor were they particularly identified as anything predominantly; the predictions were generally d6, d8, d12, or d20 (as noted above, only rarely d4). Dice outside of the d10 category were seldom identified as d10, either.

This distribution of success is not entirely surprising. In reality, d4 is arguably the most easily recognized of the categories, featuring distinctively sharp angles and a lower overall size than any of the others. d10, being the only shape that is not a Platonic solid, has perhaps the least distinctive geometry.

## Conclusion

In summary, while this simple implementation did well on training and validation, it performed relatively poorly against independent manual tests. While it did move the needle a bit over blind guessing, it did not provide reliable predictions.
