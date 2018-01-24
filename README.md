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

[image1]: ./writeup_image/distribution.png "Distribution of train dataset"
[image2]: ./writeup_image/before_normalize.png 
[image3]: ./writeup_image/after_normalize.png
[test_image1]: ./test_image/t1.png
[test_image2]: ./test_image/t9.png
[test_image3]: ./test_image/t10.png
[test_image4]: ./test_image/t12.png
[test_image5]: ./test_image/t14.png
[test_image6]: ./test_image/t20.png
[test_image7]: ./test_image/t25.png
[test_image8]: ./test_image/t28.png
[test_image8_resize]: ./writeup_image/t28_resize.png

[test_image9]: ./test_image/t29.jpg
[test_image9_problem]: ./writeup_image/t29_.png

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/HereIsPatrick/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? 34799
* The size of the validation set is ? 4410
* The size of test set is ? 12630
* The shape of a traffic sign image is ? (34799, 32, 32, 3)
* The number of unique classes/labels in the data set is ? 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I would like to use RGB(3 Channel) for inputs. so I skip the grayscale.

Only use normalize for preprocess images.

I try three different normalize algorithm.

We can see as below code block.

after train & validate, I realize that different algorithm of normalize, have wide difference.

Finally, I keep "data/255 * 0.8 + 0.1", we can see the different result of the same image as below.

First image(before) without use normalize, second(after) use normalize.

It cause the image more brightness.

And I can train in 15 EPOCH, and Validation Accuracy reach up to 0.961.


	def normalize(data):    
   		#return (data-128) / 128
    	#return data / 255 * 0.7 + 0.2
    	return data / 255 * 0.8 + 0.1

![alt text][image2]
![alt text][image3]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x6 				|
| Convolution 3x3	    | output 10x10x16      									|
| Fully connected		| Input 400, outputs 120      									|
| Softmax				| outputs 43.        									|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used

optimizer : adam

batch size : 128

epochs : 15

learning rate : 0.001

I try 20 and 30 epochs, valdation accuracy is a bit better than 15 epochs.

But New Images test is awful.

Thoughts: maybe over trainning or different shuffle

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?  96.3%
* validation set accuracy of ? 96.3%
* test set accuracy of ? 89%

Currently I learned only one neural netowrk architecture(LeNet)

So I base on LeNet architecture, modify input & output.

It's very interesting to solve image recognition without much coding.

I tuned some parameters, results are huge different.

EPOCHS is from 15 to 30. 

When 30 EPOCHS is overfiting, so test is not good. Accuracy down to 49% 

And 15 EPOCHS, Accuracy up to 89%.

Normalize algorithm is the other parameter to try.

I disover that Shuffle is important cause.

In the same parameters, each time shuffle will have different accuracy.

I do not adjust any layer (convolution, pooling, etc)


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][test_image1] 
![alt text][test_image2] 
![alt text][test_image3] 
![alt text][test_image4] 
![alt text][test_image5] 
![alt text][test_image6] 
![alt text][test_image7] 
![alt text][test_image8] 
![alt text][test_image9] 

image t29 might be difficult to classify because resize make shape fragmentary, we can see as below small picture.
![alt text][test_image9_problem] 
#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)    		| Speed limit (30km/h) 									| 
| No passing    			| No passing									|
| No passing for vehicles over 3.5 metric tons					| No passing for vehicles over 3.5 metric tons											|
| Priority road	      		| Priority road					 				|
| Stop			| Stop      							|
| Dangerous curve to the right    		| Dangerous curve to the right 									| 
| Road work    			| Road work								|
| Children crossing					| Speed limit (60km/h)											|
| Bicycles crossing	      		| Bicycles crossing					 				|


The model was able to correctly guess 8 of the 9 traffic signs, which gives an accuracy of 89%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


t1.png:

	1: 100.00%
	2: 0.00%
	6: 0.00%
	32: 0.00%
	5: 0.00%
	
t9.png:

	9: 99.99%
	28: 0.00%
	17: 0.00%
	30: 0.00%
	14: 0.00%

t10.png:

	10: 99.83%
	11: 0.17%
	42: 0.00%
	23: 0.00%
	21: 0.00%

t12.png:

	12: 100.00%
	11: 0.00%
	42: 0.00%
	6: 0.00%
	14: 0.00%

t14.png:

	14: 100.00%
	15: 0.00%
	0: 0.00%
	17: 0.00%
	3: 0.00%

t20.png:

	20: 100.00%
	23: 0.00%
	42: 0.00%
	16: 0.00%
	19: 0.00%

t25.png:

	25: 100.00%
	20: 0.00%
	23: 0.00%
	19: 0.00%
	22: 0.00%






t28.png:

	3: 88.39% -- Speed limit (60km/h)
	11: 6.32% -- Right-of-way at the next intersection
	23: 5.21% -- Slippery road
	6: 0.05%  --End of speed limit (80km/h)
	32: 0.03% -- End of all speed and passing limits

ID 28 is Children crossing, 

Look at top five possibility doesn't make sense.

check the resize of input image as below, It seems okay.

Why ID 28 is not into top five possibility.   

I have no idea for now.

![alt text][test_image8_resize] 

t29.jpg:

	29: 100.00%
	0: 0.00%
	1: 0.00%
	2: 0.00%
	3: 0.00%






### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


