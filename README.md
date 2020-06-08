# Fall-Detection_Dynamical_haar_Cascade
Knowing that Falling, like other Activities of Daily life (ADLs), is a dynamical movement, we use dynamical features to detect it in RGB cameras. Dynamical features are features that are extracted from consecutive frames of a video and hence can describe dynamical changes between frames. Although one may found different ways to extract these kinds of features (, e.g. LSTM), we use the dynamical haar cascade frame which extracts haar-like features from 3 or 4 consecutive frames. In the ["Detecting Pedestrians Using Patterns of
Motion and Appearance"](https://ieeexplore.ieee.org/document/1238422), dynamical haar features from 2 consecutive frames, and a cascade method are used to detect pedestrians in Videos. Like the aforementioned work, this work uses the haar-cascade method to detect Falls from ADLs; however, as detection of Fall needs more subtle features, we have extended the method to multiple frame numbers which can be adjusted for the data set or some specific Activities.

## Main Idea
Haar-Like features method is a fast object detection method among other detection methods. Its speed and accuracy is a motivation to use them to recognize activities in videos; however, activities have dynamical features that cannot be extracted from just one image. In this regard, it would be so beneficial if we could extract dynamical features using the haar cascade method. In the paper entitled "Detecting Pedestrians Using Patterns of
Motion and Appearance", Viola, Jones, and Snow have proposed a way to extract haar-like features from 2 consecutive frames in a video. They used these features to detect pedestrians in a camera. 

Although Viola, Jones, and Sno could gain an accurate pedestrian detection method using only 2 consecutive frames, to detect falling from other activities, we need to use more number of frames and a number of features. 

## Pre-processing

The Whole number of features that can be extracted from a video grows exponentially as the size of its frames increases, so it is important to resize the videos to have a reasonable and sufficient number of features. Following Viola, Jones, and Snow's work, the proper size is adjusted to 20*15 Pixels. 


## Training Phase
To use this code, one has to train it first. The training phase consists of extracting all possible features and deciding which feature is more important. The whole number of extracted features from three 20*15-sized frames is 700272, so this phase is very time-consuming. However, to have a fast detection algorithm, the importance of each feature is calculated by using the Adaboost Cascade method, and in the detection phase, the most important features are used. 

The results of this phase are the code which is named "final model". This code extracts only the needed features from the video. This code would be generated automatically in the training phase. 

## Detection phase and an Example
Once the "final model" code is generated in the training phase, you can use that to detect falls from other activities. "Main code" is an example of the whole process. 
 
 ## Any Quesstions?
 If you had any questions about using this code, Please contact [Amirreza Razmjoo](mailto:amirreza.razmjoo@gmail.com?subject=[GitHub]%20Fall%20Detection%20Using%20Haar%20Features )
