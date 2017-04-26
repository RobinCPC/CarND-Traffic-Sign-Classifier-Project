# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/class_hist.png "Histogram of each class in training set"
[image2]: ./examples/random_show.png "Random Showing"
[image3]: ./examples/preprocess.png "Preprocessing image"
[image4]: ./examples/lenet.png "LeNet"
[image5]: ./examples/train_valid.png "Training Process"
[image6]: ./examples/ten_sample.png "Testing Samples"
[image7]: ./examples/sample1.png "Sample 1"
[image8]: ./examples/sample2.png "Sample 2"
[image9]: ./examples/sample3.png "Sample 3"
[image10]: ./examples/sample4.png "Sample 4"
[image11]: ./examples/sample5.png "Sample 5"
[image12]: ./examples/sample6.png "Sample 6"
[image13]: ./examples/sample7.png "Sample 7"
[image14]: ./examples/sample8.png "Sample 8"
[image15]: ./examples/sample9.png "Sample 9"
[image16]: ./examples/sample10.png "Sample 10"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

###Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3 (RGB channel)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how how many images in each class in training set.
From the histogram, some classes have very few data, it will make some bias to the classes with more data.
![alt text][image1]

I also randomly show couple images in training set.
![alt text][image2]
 
###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to YUV color scope because Y channel is weighted combination of RGB channel. Y channel, also one kind of grayscale, can preserve the features of shape in RGB channel. And, color do not play important role in traffic sign classifier. 
As a last step, I normalized the image data because, as in class, normalize the features (dimension) can be more easy to train with single learning rate. After normalized, each feature (pixel) will similar distriubtion and one learning rate can fit to all features.

Here is an example of a traffic sign image before, after grayscaling and normalizing.

![alt text][image3]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I LeNet model architecture with more weights and I also add dropout regulator to make my neural network more robust.
![LeNet Model][image4] 
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale (Y channel) image   		| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x15 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x15 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x40 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x40 				    |
| Fully connected 1		| Flatten previous layer, output 1000 features single layer |
| Dropout layer         | dropping rate 50%                             |
| Fully connected 2		| 1000 in 300 out       						|
| RELU					|												|
| Dropout layer         | dropping rate 50%                             |
| Fully connected 3		| 300 in 210 out        						|
| RELU					|												|
| Dropout layer         | dropping rate 50%                             |
| Output layer			| reduce to final 43 features   				|
| Softmax_CrossEntropy  | loss layer (with one hot encoded)				|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer (bult-in in TensorFlow), my final setting as following:

    * Batch size: 128
    * number of epochs: 60
    * mean (mu): 0.
    * std. (sigma): 0.1
    * dropout (keep probability): 0.5

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100 %
* validation set accuracy of 97.6 %
* test set accuracy of 96.9 %

The following is epoch vs accuracy of training and validation.
![alt text][image5] 

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The first I tried, which is suggested in class, is LeNet for hand writting. I change input with 3 channel (not 1 in hand writting) for RGB images.
* What were some problems with the initial architecture?
The model run well but it only get 90 % in validation accuracy.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Because the original LeNet have shallow filter depth (6 in first layer, 16 in second layer) for 10 hand writing classes, I increase the filter depth of each layer (15 in first layer, and 40 in second layer).
In addition, in order to make the model more robust to dataset that may not have all features (nosie on images). I add dropout layer in full connected layer to randomly ignore certain mount of features. The dropout rate I choose is 50 %.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web:

![alt text][image6] 

The only uncorrect is the first image "double curve". Because the sign I get is actually different from dataset. The "double curve" in dataset is turn left first then turn right, but the sighn I get is turn right first. Other images are correctly calssified.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Double curve   		| Children Crossing   							| 
| Right-of-way    		| Right-of-way									|
| 30 km/h				| 30 km/h										|
| 60 km/h				| 60 km/h										|
| Priority Road			| Priority Road      							|
| General Caution		| General Caution				 				|
| Road work             | Road work				 	                	|
| Turn left ahead       | Turn left ahead		 	                	|
| Bumpy Road			| Bumpy Road        							|
| Keep Right            | Keep Right                                    |
|                       |                                               |
|                       |                                               |


The model was able to correctly guess 9 of the 10 traffic signs, which gives an accuracy of 90%. This compares favorably to the accuracy on the test set of 96.9%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the 1st image, the model is relatively sure that this is a children crossing sign (probability of 0.58), but, as I said, this "double curve" sign is diffrent (mirror images) from the train dataset
![alt text][image7]

For the 2nd image, the model is relatively sure that this is a Right-of-way at next intersection sign (probability of 1.0)
![alt text][image8]

For the 3rd image, the model is relatively sure that this is a speed limit (30 km/h) sign (probability of 1.0)
![alt text][image9]

For the 4th image, the model is relatively sure that this is a speed limit (60 km/h) sign (probability of 1.0)
![alt text][image10]

For the 5th image, the model is relatively sure that this is a priority road sign (probability of 1.0)
![alt text][image11]

For the 6th image, the model is relatively sure that this is a general caution sign (probability of 1.0)
![alt text][image12]

For the 7th image, the model is relatively sure that this is a road work sign (probability of 1.0)
![alt text][image13]

For the 8th image, the model is relatively sure that this is a turn left ahead sign (probability of 1.0)
![alt text][image14]

For the 9th image, the model is relatively sure that this is a bumpy raod sign (probability of 1.0)
![alt text][image15]

For the 10th image, the model is relatively sure that this is a keep right sign (probability of 1.0)
![alt text][image16]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


