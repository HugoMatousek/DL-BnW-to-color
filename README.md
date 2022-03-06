# BnW to color #
## A Deep Learning model that turns white and black images to color images ##

Just as the subtitle says, the aim of this project is to create a deep-learning-based model that will colorify black and white images.

The basic idea is to use LAB channels. Namely, predict the A and B channels from the L channel. In the later versions of the model, I plan on implementing some sort of image classification that will help the model to understand the contents of the image to improve the colorification.

Several models and datasets will be used.

As of right now, the project is ready to download and preprocess data. Namely, it downloads a dataset of landscape images and flicker8k. At the moment, it outputs X and Y matrices (0-255 floats) consting of the corresponding LAP channels in the correct shape.

By Hugo Matousek
