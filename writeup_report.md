## Writeup Report
### This is Chuan's writeup report file for Udacity Self-Driving Car nanodegree term1 project 5 Vehicle Detection and Tracking

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.jpg
[image2]: ./output_images/HOG_Example_1.jpg
[image3]: ./output_images/SmallScaleFactor.jpg
[image4]: ./output_images/Test6ScaleFactor.jpg
[image5]: ./output_images/combinedboundingboxes.jpg
[image6]: ./output_images/heatmap.jpg
[image7]: ./output_images/UpdatedHeatMap.jpg
[image8]: ./output_images/AllTestImagesCarDetect.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how I addressed each one.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the G channel in `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and here are the results for training these HOG features with different combinations of parameters using Linear SVC (C=1):

| Parameter Configurations        |  Linear SVC (C=1) Training Results  | 
|:-------------:|:-------------:| 
| ColorSpace: RGB; Orient: 9; PixPerCell: 8; CellPerBlock: 2; HOG Channel: ALL   | Time: 44.79s; Accuracy: 0.922         | 
| ColorSpace: HSV; Orient: 9; PixPerCell: 8; CellPerBlock: 2; HOG Channel: ALL   | Time: 31.87s; Accuracy: 0.9555         | 
| ColorSpace: LUV; Orient: 9; PixPerCell: 8; CellPerBlock: 2; HOG Channel: ALL   | Time: 22.31s; Accuracy: 0.9665         | 
| ColorSpace: HLS; Orient: 9; PixPerCell: 8; CellPerBlock: 2; HOG Channel: ALL   | Time: 28.45s; Accuracy: 0.9533         |
| ColorSpace: YUV; Orient: 9; PixPerCell: 8; CellPerBlock: 2; HOG Channel: ALL   | Time: 21.52s; Accuracy: 0.9651         | 
| ColorSpace: YCrCb; Orient: 9; PixPerCell: 8; CellPerBlock: 2; HOG Channel: ALL   | Time: 21.03s; Accuracy: 0.9685         | 

With above results, I decide to use YCrCb color space which provides a highest test accuracy and least training time. Then I try to optimize other parameters with the same trial and test method, finally I decide to use this parameter combination:

**ColorSpace: YCrCb; Orient: 15; PixPerCell: 16; CellPerBlock: 2; HOG Channel: ALL**

With this I can bump up the test accuracy to **0.9786**, and the training time is reduced to only **4.78s**.

Besides that, I also tried to combine color spatial and histogram features. And it turned out that the accuracy is increased to **0.9887**, but with training time increased a little bit to **11.26s**.

I think this is good result.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I have trained a linear SVM classifier using above mentioned HOG features plus color spatial and histogram feataures. The code is in 12th cell in ipynb file [Project5_Pineline.ipynb](https://github.com/GitHubChuanYu/Project5_VehicleDetectionAndTracking/blob/master/Project5_Pipeline.ipynb). And I tried to tune the parameter C for linear SVM and found out a smaller C value (0.001) would give a higher test accuracy to **0.9904**.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Firstly, I pick up the **find_cars** function in the class slide 'Hog Sub-sampling Window Search' which allows me to do sliding window search with different scale. The base window size and overlap percentage is the default value from the class. The base window size is 64x64, and overlap percent is controlled by parameter cells_per_step, which is 2. And pixel per cell is 16, so the total cells are 4x4. So cells_per_step is 2 means overlap percentage is (4-2)/4 = 50%. However the overlap percentage is changing if the scale factor does not equal to 1. 

Secondly, I am using this function to test on different output images from the project video and also test images. Basically the principle is to use small scale factor when the car is far way and close to horizontal line while using large scale factor when the car is close to the camera. Also the search window (y_start & y_stop) can be adjusted based on different car locations in different images.

For example, here I limit the searching area to a small area close to far away horizontal line (**ystart = 405**, **y_stop = 490**), and use a small scale factor as **1.1** and test on one video image, the result is pretty good with a fit box to cover the car:

![alt text][image3]

And in contrast, I increas the seaching area to a large area close to camera (**ystart = 405**, **y_stop = 600**),  and use a large scale factor as **2** and test on one video image, the result is pretty good with several large overlapped boxes to cover the car:

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

You can see above two examples to demonstrate that the **find_cars** function is working to detect cars located differently in different images with using different searching area and different scale size. One more example using [test6.jpg](https://github.com/GitHubChuanYu/Project5_VehicleDetectionAndTracking/blob/master/test_images/test6.jpg) is shown here:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here is the link to my video output [project_video_output_mem.mp4](https://github.com/GitHubChuanYu/Project5_VehicleDetectionAndTracking/blob/master/project_video_output_mem.mp4).


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

Here's an example result showing the image of combined bounding boxes for car, heatmap of the combined bounding boxes, updated heatmap after applying threshold, and updated heatmap repplied to the original image for identifying cars.

### Here is image with original bounding boxes of multiscale sliding window search:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap:
![alt text][image6]

### Here are the resulting bounding boxes are drawn onto original image and updated heatmap with threshold applied to filter multidetections and false positives:
![alt text][image7]

### Here are final results of all test images with applied heatmap thresholding filters:
![alt text][image8]

For the final processing of project video, I have also applied a techinque to combine and average the heatmaps of several conseutive frames in the video, this is helping to filter out false positives further and also to make the detection of cars more stable and consistent while the car is moving. The code is in function **image_processing_mem**, the code is displayed here:  
```Python
if len(history.history) >= 7:
    history.history = history.history[1:]
    
history.history.append(heat)
heat_history = reduce(lambda h, acc: h + acc, history.history)/7
    
heat = apply_threshold(heat_history, 3)
```
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

During implementation of this project, the main problem I have is to compromise between getting rid of false positives and getting fitting and stable detection boxes of cars. If I reduce the heatmap threshold value, then I can get better fitting and detection boxes of car but also have more false positives. However if the heatmap threshold value is large, then I have less false positives but with not very good fitting boxes (always small and sometimes none) to detect cars. 

If I have time in the future, I am interested to learn new ways to automatically find the better compromised parameters to both get rid of false positive effectively and also have a better fitting boxes to detect cars.
