# **BrainTumorDetection**

## **Data**
The data was collected from [Kaggle](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset)
There are 4600 instances with 55% of them has brain tumor and 45% of them are healthy. The images have various sizes which will then be converted into a 100x100 pixels for CNN model and 256x256 for ResNet as suggested in [PyTorch documentation](https://www.pytorch.org/hub/pytorch_vision_resnet/).

## **Context**
The motivation behind this project is to create a Deep Learning model that is able to detect whether a brain is healthy or not. If this model is accurate in detecting brain tumors, it can be a huge advantage in helping medical teams in detecting brain tumors despite the size or stages.

## **Model**
### CNN Model
Model uses CNN method with Max pooling, Batch Normalization, Dropout techniques to prevent overfitting due to it's fairly complex characteristics. It then uses 2 fully connected layers as classifiers. There are a total of 8 hidden layers, in which 6 of them uses **Convolutional Neural Network** technique. Please see below for its hyperparameters and accuracy

Training Hyperparameters for the CNN model:
* Learning Rate = 0.01
* Batch Size = 128
* Weight Decay = 0.00001
* Dropout Probability= 0.4

Final model:

![BrainTumorDetector](https://github.com/mart1428/BrainTumorDetection/blob/main/images/BrainTumorDetection_TrainVal.png)
![BrainTumorDetector](https://github.com/mart1428/BrainTumorDetection/blob/main/images/BrainTumorDetection_Test.png)

### ResNet - Transfer Learning
For this model, ResNet18 model is taken from ```torchvision```. This model, as its name suggested, has 18 layers and it has been trained for its optimal weights. The "Feature Extraction" parts were then taken and used for the current Brain Tumor Detection problem. Additionally, 3 fully connected layers with size 2048 were used as classifier. Please see below for more details.

Training Hyperparameters for ResNet model:
* Learning Rate = 0.0001
* Batch Size = 128
* Weight Decay = 0.00001
* Dropout Probability= 0.5

Final model:

![ResNet18](https://github.com/mart1428/BrainTumorDetection/blob/main/images/ResNet18_TransferLearning_TrainVal.png)
![ResNet18](https://github.com/mart1428/BrainTumorDetection/blob/main/images/ResNet18_TransferLearning_Test.png)

Interestingly, the CNN model that I created did better than transfer learning. Although it is not a 1 to 1 comparison due to the different Image Sizes, there is a high chance that ResNet needs more time in fine tuning to obtain a better result.

## **Future Research**
Further considerations to be made in the future:
* More data points.
* Data augmentation techniques such as rotating and flipping.
