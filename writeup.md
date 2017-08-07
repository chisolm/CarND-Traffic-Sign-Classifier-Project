#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


# TODO add images
[//]: # (Image References)

[image1]: ./writeup_examples/label_histogram.png "histogram"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./writeup_examples/hist_rgb.png "Histogram RGB levels"
[image4]: ./writeup_examples/learning_rate_1.png "Learning rate"
[image5]: ./writeup_examples/learning_rate_2.png "Learning .005"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

[imagets1]: ./test_images/2764561332.1.png "Road narrows right"
[imagets1]: ./test_images/2764561332.2.png "Bumps"
[imagets1]: ./test_images/Achenpass.png "Bicycle crossing"
[imagets1]: ./test_images/Road_Sign_And_Winter_Scenery.png' "Double curve"
[imagets1]: ./test_images/Umleitung_Sackgasse_Anlieger_Frei_Baustelle_Pullach_im_Isartal.3.png "Leaves 30km/h"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Link to Chris Chisolm's (me) [project code](https://github.com/chisolm/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?  34799
* The size of the validation set is ? 4410
* The size of test set is ? 12630 
* The shape of a traffic sign image is ?  (32, 32, 3)
* The number of unique classes/labels in the data set is ? 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. 

I did very limited pre-processing of the data.  I normalized it to a -1 to 1 range.  I experimented with gray scale early in my
development but did not see a benefit in my use.  This experiment was before I added an regularization.  At the time my model was 
overfitting significantly on the color version, I suspect the black and white so no significant gains because it was subtracting 
information and may have exagerated the overfitting, or results were lost in the noise.

I decided not to generate additional data because one of the techniques suggested was to jitter the image.  Looking at the
images in the training set, it appears that they may have been taken from a moving vehicle based on background changes.  These
are likley to already resemeble a jitter type of augmentation.

Augmentation by translating the apparent angle of the sign could still be useful.  I would like to experiment in the future
with occlusion of parts of the image.  I wonder if this with have a dropout like effect forcing the network to develop 
redundant recognication.

Brightness augmentation also may provide useful cases.  The data set has the appearance of low contrast/low intensity.

![alt text][image3]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I kept my model small since I was originally developing and testing it on my mac laptop.  There was also signs that
even the basic lenet homework exercise model was overfitting for the data in the training set.  This did not encourage
increasing the model size.  

The reduced size also greatly aided the exploration of hyperparamter changes and exploration of multiple dropout
combinations.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     		| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU				|												|
| Max pooling	      		| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    	| 1x1 stride, valid padding, outputs 10x10x16
| RELU				|												|
| Max pooling	      		| 2x2 stride,  outputs 5x5x16 				|
| Flattening layer      	| 2x2 stride,  outputs 400 				|
| Fully connected		| 400x120.        									|
| Optional dropout layer 	| keep_prob = parameter.        									|
| Fully connected		| 120x84        									|
| Optional dropout layer 	| keep_prob = parameter.        									|
| Fully connected		| 84x43        									|
| Softmax			|      									|
|				|												|
|				|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I chose a minimization of the mean softwax for my optimizer.  Originally I varied the number of epochs, but based on a design choice to
minimize turn around time on a slower machine, I settled on a value of 20 for the number of epochs I used.  I could have further benefitted by
creating an early exit capability.  This did affect my choices on learning rate as rates at or below 0.0001 failed to finish training within
20 epochs.  

I ran a number of tests sets that varied

hyperparam['dropout'] = 'fc5fc6'
hyperparam['conv'] = '5x5'
hyperparam['EPOCHS'] = 20
hyperparam['BATCH_SIZE'] = 64
hyperparam['learning_rate'] = .0005
hyperparam['keep_prob'] = 0.7

run_batch = False

| Hyperparameter 	| Options         		|
|:---------------------:|:---------------------------------------------:|
| Epochs         		| 20				| 
| Batch Size         		| 64, 128, 256			| 
| Learning rate         	| .005, .001, .0005, .0001	| 
| Dropout         		| none, fc6 only, fc5 and fc6	| 
| Keep Probability         	| 0.5, 0.6, 0.7			| 

The following graphs strongly suggest a learning rate of 0.0001 is too small. None of the 0.0001 runs converge to a steady loss by epoch 20. It also suggests an adaptive learning rate would be useful as 0.005 starts very well for the first 2-5 epochs.

The format includes all runs in gray with a low alpha to allow visibility.

![alt text][image4]

The following shows the instability in the 0.005 learning rate.

![alt text][image5]

The higher accuracy models were found with the following hyperparaters sorted by validation accuracy:

|Accuracy    |  Dropout, batch size, learning rate, epochs, keep_prob |
|:---------------------:|:---------------------------------------------:|
|0.9621315190310922	| 'fc5fc6', 64, 0.001, 20, 0.5	|
|0.9621315187878079	| 'fc6', 64, 0.001, 20, 0.6	|
|0.9619047619317935	| 'fc5fc6', 64, 0.0005, 20, 0.7	|
|0.96122448982295	| 'fc5fc6', 64, 0.0005, 20, 0.7	|
|0.9596371882086168	| 'none', 64, 0.001, 20, 0.6	|
|0.9596371879653325	| 'none', 128, 0.001, 20, 0.7	|
|0.9591836732261035	| 'fc5fc6', 64, 20, 0.5, 0.001	|
|0.9589569160997733	| 'fc5fc6', 256, 0.001, 20, 0.7	|
|0.9589569158835206	| 'none', 256, 0.005, 20, 0.7	|
|0.9587301585139061	| 'fc5fc6', 64, 0.001, 20, 0.6	|

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

I started with the lenet network that we used in class.  The rough relationship of image input size from the lenet example suggested to me that 
it could at least come close to the solution I was looking for.  I expected to have to increase the size of the fully connected layers at least
to cope with the increase complexity in the images.  I also suspect that I would need to do something with the convolution layers, but I did
not know what.  I used RELU activation for all layers.

As I started testing I kept seeing consistent signs of overfitting due to the decreasing loss on the training set not being followed by
the validation loss.

I ended up with nearly the same architecture.  The only change I made was to add an option to use dropout.  I made that an option
so I could experiment with dropout configurations.

My final model results were:
* training set accuracy of ? 99.5%
* validation set accuracy of ? 96.2%
* test set accuracy of ? TODO

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

Image 1: The road narrows is difficult to resolve even for a human:

![alt text][imagets1] 

Image 2: The bump on 'bumpy road' is only 1 pixel high

![alt text][imagets2] 

Image 3: Bicycle crossing, this image was frequently mis-identified on other tuning values.  The crossing was always identified, the bicycle was frequently interpreted as pedestrians.

![alt text][imagets3] 

Image 4: This is again difficult due to the need to resolve a symbol in the center.

![alt text][imagets4] 

Image 4: This speed symbol is significantly abscrued with leaves.

![alt text][imagets5]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        | 
|:---------------------:|:---------------------------------------------:| 
| Road narrows on right      	| Road narrows on right   	| 
| Bumpy road     		| Bumpy road 			|
| Bycycles crossing		| Bycycles crossing		|
| Double curve 	      		| Children crossing  		|
| Speed limit 30 km/h 		| Speed limit 30 km/h 		|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. I have an addition 5 signs not pictured here but included in the 
notebook, the accuracy is only 6 out of 10 for 60%.  This compares poorly to the validation set accuracy of 96.2% and test set accuracy of TODO.
 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 96.6% 			| Road narrows on right   	| 
| 99.8% 			| Bumpy road 			|
| 99.8% 			| Bycycles crossing		|
| 80.1%				| Double curve 	      		| 
| 98.7%  	 		| Speed limit 30 km/h 		|

For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


