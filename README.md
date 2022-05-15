# Image colorization using convolutional autoencoder and LAB color space

This project uses two deep learning models based on the autoencoder architecture to color grayscale (black & white) images. It was done solely for educational purposes as a part of the [Deep Learning course](https://www.ait-budapest.com/syllabuses/deep-learning) at [AIT Budapest](https://www.ait-budapest.com/).



## Introduction
Image colorization is an interesting topic for machine learning and for deep learning in particular. There is a huge amount of historical pictures and videos that were taken before the invention of colored film and probably even more material created in recent years in black & white for whatever reason. Normally, coloring a black & white image would take hours of manual work, even for an experienced graphics expert. However, with deep learning and neural networks, this process can be fully automated with results that are comparable to human-done colorization. For example, already in 2016, a deep neural network bot was deployed on Reddit, where it fooled many fans of historical photography on the [r/oldSchoolCool](https://www.reddit.com/r/OldSchoolCool/) subreddit.

I chose the black & white image colorization as my term project because it is a nice demonstration of deep neural networks' direct deployment to produce results that were unimaginable to automate just a couple of years ago. Furthermore, while it seems relatively straightforward, there are many possible approaches and architectures to choose from and many technical obstacles to overcome. 



## Previous solutions
There are tens, if not hundreds, of full solutions to the colorization problem. They all vary, but there are six major approaches that are most commonly used [\[1\]](https://emilwallner.medium.com/colorize-b-w-photos-with-a-100-line-neural-network-53d9b4449f8d). Namely, they are:

 1. Simple 2d convolutional autoencoder
 2. Guiding the neural network by manually adding small dots of color
 3. Transfer coloring from a similar image
 4. Using residual encoder and merging classification layers
 5. Using hypercolumns from classifying network
 6. Infusing the autoencoder with the outcome of classifying the network

The first method simply uses 2D convolutional layers with some stride setting to compress the input and maximize information without distorting the image and then use upsampling layers to return to the desired resolution. This architecture tends to work well for small resolutions and for datasets of similar content. An example can be found [here](https://www.geeksforgeeks.org/colorization-autoencoders-using-keras/).

The second solution is based on the premise that neighboring pixels tend to have similar colors and intensity. It requires human interaction with the input images, so I did not explore it further. However, the results seem to be impressive, and the technique avoids many problems with the stability of coloring (important for video coloring) and even allows recolorization of segments of already colored images. Click [here](https://www.cs.huji.ac.il/w~yweiss/Colorization/) for more details and see the example below:


<details>
 <summary>Image colorization using dots - example visualization</summary>

 ![Image colorization using dots - example](https://i.imgur.com/eWkCiy9.png)
</details>

As the name suggests, the third approach also needs some additional input from a human operator in the form of a reference image. Because of that, I did not explore this further. However, feel free to see more details [here](https://dl.acm.org/doi/10.1145/2393347.2393402).

The fourth method uses an interesting deep neural network architecture that uses a pre-trained model and lets layers interact in a non-sequential manner. The model's architecture can be seen in the image below, and details can be found [here](https://tinyclouds.org/colorize). (please use the light theme to see the image properly)

<details>
 <summary>Residual model - architecure visualization</summary>

 ![Residual model - architecure](https://tinyclouds.org/colorize/residual_encoder.png)
  
</details>

Using the hypercolumns (i.e., the fifth approach) works on the principle of effectively building a 2D convolution image classifier and building it into a hypercolumn that helps determine the channels. An interesting project using this technique can be found [here](https://github.com/BerenLuthien/HyperColumns_ImageColorization).

The last approach is somewhat a combination of the first one and the fifth one. It uses the autoencoder architecture, but it adds an output from a pre-trained classifier network to the bottleneck. In effect, this should give the autoencoder more information about the content of the image and help it to color it better. See the following [link](https://arxiv.org/pdf/1712.03400.pdf) for more details and take a look at the idea behind the model below. (please use the light theme to see the image properly)


<details>
 <summary>Autoencoder with inception input visualization</summary>

 ![autoencoder with inception input](https://miro.medium.com/max/1400/1*KRXxAAxlBz1psRvB1ak04Q.png)
  
</details>

Finally, as I discovered from those previous implementations, the color space used is important. While it is possible to directly RGB color an image, this requires the network to guess 3 different channels. Therefore, other color spaces such as Lab, YUV, HSV, and LUV are often used. Their advantage is that only two of their channels bear colors while one of them is effectively the grayscale version of the image, so only two channels need to be guessed.
## Datasets
Given that colorization is very dependent on the content of the images that we want to color, I decided to use two datasets where one of them consists of the same type of content, and the other is varied. I experimented with several datasets, but in the end, I decided to use the Labeled Faces in the Wild (LFW) dataset from the University of Massachusetts Amherst [\[2\]](http://vis-www.cs.umass.edu/lfw/) and flicker8k dataset obtained from Kaggle [\[3\]](https://www.kaggle.com/datasets/jainamshah17/flicker8k-image-captioning). 

The LFW dataset consists of 13233 images of the faces of 5749 publicly known people. At the same time, the images vary in angle, size, background, and other details. Because of those, I considered it a good dataset that has the same content in the pictures but would be less susceptible to overfitting.

On the other hand, the flicker8k dataset consists of 7999 images of various content. As such, it was a perfect candidate for the dataset that I wanted to be varied.

## Proposed method
I decided to implement two different models. The first one is the simple 2d convolutional autoencoder model. The second one is partially based on the first one but includes an additional "fusion" part in the autoencoder's bottleneck that merges the outcome of a pre-trained image classifier network into it. 

I decided to use the LAB color space where `L` is the lightness (essentially the grayscale part of the image), and `a` and `b` are the green-red and blue-yellow color spaces, respectively. The individual channels are exemplified in the image below [1]:

<details>
 <summary>LAB channels visualization</summary>
 
 ![LAB color space](https://miro.medium.com/max/1400/1*OX9DWIK6bOHKwTAp4Q92pQ.png)
</details>

This way, the models only need to predict the `a` and `b` channels.

The next decision was the resolution. The higher the resolution, the greater requirements on computational resources and also on-time needed to train the models. In the research that I have done, I found out that with small resolutions, even small and simple models do well. However, I wanted models to be actually useful, so I did not want to settle for a resolution too low. Additionally, given the convolutional nature of my models, I needed the resolution to be easily divisible by 2 and its powers. All this considered, I settled with the `224x224` resolution, which I knew should work based on the previous solutions. Finally, since I wanted to compare both models, I used the same resolution for both.

As a part of the preprocessing, I made sure that the images were resized to this resolution. I also cropped them to keep to avoid image distortion, if needed, but that was probably not crucial.

### Model 1
As already mentioned, the first model is a 2D convolutional autoencoder. I tried different variations, a different number of the convolutional layers, different parameters for each layer, and different activation functions and optimizers. In the end, the model architecture in the image below proved to be the best. For hidden layers, I used `relu`, and for the last layer, I used `tanh` as the `ab` channels have values in the `-128 to 128` range. As the optimizer, `Adam` proved to be the best thanks to its adaptive learning rate. The model has 4,756,770 trainable parameters. 

<details>
  <summary>Model #1 architecture</summary>
 
  ![model 1](https://github.com/HugoMatousek/DL-BnW-to-color/blob/main/1_model.png?raw=true)
</details>

### Model 2
I decided to keep the most from the first model in the second one. However, I changed some of the layers and added new ones. The most significant change, however, is the addition of the fusion layer to the bottleneck of the autoencoder. I have tried a could of pre-trained networks (such as `InceptionV3` or `MobileNet`) but ended up using `InceptionResNetV2`. In the final version of the model, I take the input image and use its `224x224` version as an input for the encoder part of the model. At the same time, however, I also resize the image to `299x299` (InceptionResNetV2 default resolution), repeat the vector three times (as the model requires three channels), preprocess it accordingly, and send it to the `InceptionResNetV2` model. In the fusion layer, I take its output, and after repeating the vectors according to the needed resolution, I reshape it and merge it with the encoder output. This is then convoluted and sent to the decoder. I tried variations of the fusion layer, but most of them proved to either completely destroy the model or to provide no benefit. The `InceptionResNetV2` part of the network is not trainable, so the resultant model has 7,075,922 trainable parameters. You can see its architecture below.

<details>
  <summary>Model #2 architecture</summary>
 
  ![model 2](https://github.com/HugoMatousek/DL-BnW-to-color/blob/main/2_model.png?raw=true)
</details>

## Evaluation method
In general, looking at just accuracy or loss values is always tricky with autoencoders. This is especially true for image colorization, where we care about the overall result rather than a loss in individual pixels. Moreover, there are various aims that we might have with image coloring. Namely, it is important to distinguish whether we want to recover the true original colors or just want to have a sensible colored version of a previously black & white image. In most cases, the latter is true. Then, coloring an originally red t-shirt green is more desired compared to having it brownish with some rainbow artifacts, even though it will produce higher loss values. Consequently, while I still use the Keras evaluation function for both models, its outcomes are secondary. Rather than that, I color 50 images from a test set that the network has not seen before and do a subjective evaluation. Of course, such evaluation is somewhat limited, but there is no other feasible alternative. Finally, since the aim of image colorization is usually to color images that only exist in the black & white version (no available ground truth), I also decided to evaluate the models based on the results from doing so.



## Results
I trained both models with different combinations of the two datasets (see more below). Generally, I used a batch size of 48 with 1920 images in the training and validation set (80/20 split). While I tried different batch sizes, especially for testing and later fine-tuning, the batch size of 48 seems to have the best results/time ratio. Additionally, while 1920 images are just a fraction of both of the datasets, I decided to use this subset because of performance issues and time constraints when training the model. Given the resolution I chose, the whole process is very computationally demanding. Additionally, processing the images is very RAM-intensive. While I optimized the process, it still takes a lot of RAM. One solution would be to load the images straight from folders when training the model, but this would make the preprocessing for the second model cumbersome. 

Given the need for subjective evaluation and potentially misleading accuracy and loss values, I only used EarlyStopping to prevent a serious increase in loss value (which I read could be a problem in colorization and `Adam` optimizer - however, I never came across this problem). Additionally, I was recommended to adjust the learning rate via `callbacks`. However, I observed no significant improvement in results, so I only used it for fine-tuning.

I tried to train Model #1 with the flicker8k dataset, but the results were unsatisfying - mostly brownish. This was likely caused by a huge variety of the dataset. Combining the two datasets was a bit better, especially for images with people on them, but still very unsatisfying. However, training Model #1 with just the LFW dataset proved to work very well! See yourself:

<details>
  <summary>Model #1 test output</summary>
 
  ![1_test_output](https://github.com/HugoMatousek/DL-BnW-to-color/blob/main/1_test_output.jpg?raw=true)
</details>

Model #2, when trained on just the LFW dataset, produced comparable results to Model #1. However, it also produced decent results even with the flicker8k dataset. Finally, the best results were achieved by combining the two datasets. In this case, it is clear that the model does the best job with faces but is also capable of sensibly coloring other scenes:

<details>
  <summary>Model #2 test output</summary>
 
  ![2_test_output](https://github.com/HugoMatousek/DL-BnW-to-color/blob/main/2_test_output.jpg?raw=true)
</details>


Finally, a couple of images that were originally taken in black and white (i.e., no available ground truth):

<details>
  <summary>Model #1 historical output</summary>
 
  ![1_hitroical_output](https://github.com/HugoMatousek/DL-BnW-to-color/blob/main/1_hitroical_output.png?raw=true)
</details>

<details>
  <summary>Model #2 historical output</summary>
 
  ![2_hitroical_output](https://github.com/HugoMatousek/DL-BnW-to-color/blob/main/2_hitroical_output.png?raw=true)
</details>




## Discussion
While many of the colored images are not usable, there is a great deal of them that look actually really good - and this is the case for both models. Model #1 is obviously only capable of coloring images of similar content that it was trained on. However, given that the LFW dataset has pictures taken from many angles with various backgrounds and positions of the face, it seems that it doesn't work just as a static filter that would add skin colors to the middle of the image. I find it quite amazing that such a simple model can actually do such a good job!

With the second model, the results are slightly worse, despite its more advanced architecture. On the other hand, it is capable of somewhat properly coloring various kinds of images and scenes. Also, it is important to realize that the InceptionResNetV2 with the ImageNet weights recognizes 1000 categories and flicker8k has only 7999 images (and only about half was actually used for training). Therefore, it is likely to assume that with a better training set, it could probably do an even better job. However, even in the current state, it is obvious that the model correctly recognizes what is in the picture in most cases and colors it accordingly (for example, this is clear with images featuring grass or water bodies).

Overall, there is definitely great potential in using deep neural networks for image coloring. One of the best examples is the following [project](https://github.com/jantic/DeOldify).

Even more importantly, however, the color imaging problem is a great one for learning more about neural networks and deep learning. Even though the many problems I faced while developing this project pushed my solution closer to the previously existing solutions, it was still an extremely important and useful learning experience.


## Further References (in addition to in-text links)

[1] Wallner, E. (2021, February 9). _How to colorize black & white photos with just 100 lines of neural network code_. Medium. https://emilwallner.medium.com/colorize-b-w-photos-with-a-100-line-neural-network-53d9b4449f8d

[2] [Gary B. Huang](http://vis-www.cs.umass.edu/~gbhuang), Manu Ramesh, [Tamara Berg](http://research.yahoo.com/bouncer_user/83), and [Erik Learned-Miller](http://www.cs.umass.edu/~elm).  
**Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments.**  
_University of Massachusetts, Amherst, Technical Report 07-49_, October, 2007.  [[pdf]](http://vis-www.cs.umass.edu/lfw/lfw.pdf)

[3]  
+ Shah, J. (2020, August 20). _Flicker8k - Image Captioning_. Kaggle. https://www.kaggle.com/datasets/jainamshah17/flicker8k-image-captioning
+ M. Hodosh, P. Young and J. Hockenmaier (2013) "Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics", Journal of Artifical Intellegence Research, Volume 47, pages 853-899 http://www.jair.org/papers/paper3994.html


