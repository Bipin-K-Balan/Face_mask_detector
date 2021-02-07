# Face_mask_detector

This is a Tensorflow based object detection(TFOD) model which used to detect whether a person has wearing a mask or not. This is a personal project which inspired from the current apandemic situation.
The project is designed to either use as a real time model that instantly detect face mask from straming video or webcam or we can use it as a static model that detect the mask when you upload photo in Web browser.

### Dataset
Data set is collected using Image scapper that was created to scrap images of people who wearing face mask and write it into a directory path. Based on the availability of images in the internet our scapper only able to collect near 900 images which is not sufficient for an high performing object detection model. But due to hyperparameter tuning and increasing iterations and selecting good performing SSDlite algorithm, the model managed to give a moderate accuracy on test data set

### Training
The entire model has trained in Paperspace cloud using NVIDIA Quatro P5000 GPU ,it took around 6 hours to train for 60,000 steps with initial learning rate of 0.004.

### Object Detection Model
I am using TFOD v 1.3 frame work and using ssdlite_mobilenet_v2 in TFOD model zoo which is pretrained on COCO dataset which provides a good tradeoff between mean average precision and speed compared to other pretrained models. Means it can give better prediction maintaining better fps (can be used in real time predictions).

### Coding
Python is used for coding and few modules of TFOD 1.3 has been taken for getting the bounding boxes and instead of tensorify operation few codes has been changed interms of numpy for simplicity.
Code has been created for both real time streaming video or webcam using Open CV library and also created Web API using flask for the static prediction of facemask using images.

### Output sample

![GitHub Logo](https://ibb.co/h7H36Fw)
Format: ![Alt Text](url)
https://ibb.co/h7H36Fw
