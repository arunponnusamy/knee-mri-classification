# Knee MRI Classification with CNN, TensorFlow and Python
![](images/kneeMRI.jpeg)

Classification of Knee MRI images to recognize the Anterior Cruciate Ligament (ACL) condition.
1. healthy, 
2. partially injured, or 
3. completely ruptured

## Dataset
KneeMRI dataset was gathered retrospectively from exam records made on a Siemens Avanto 1.5T MR scanner, and obtained by proton density weighted fat suppression technique at the Clinical Hospital Centre Rijeka, Croatia, from 2006 until 2014. The dataset consists of 917 12-bit grayscale volumes of either left or right knees.

![](images/knee_mri_dataset.png)

## Improvements 
Below are some of the work to be done to improve the model performance. 
- Train EfficientNet model with TensorFlow
- Handle class imbalance with augmentation
- Try class weighting approach during training to tackle class imbalance
- Fix gradcam issue (disconnected graph)
- Explore MRNet 
