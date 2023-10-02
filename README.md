# **BrainTumorDetection**

## **Data**
The data was collected from Kaggle. Link: https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset
There are 4600 instances with 55% of them has brain tumor and 45% of them are healthy. The images have various sizes which will then be converted into a 100x100 pixels.

## **Context**
The motivation behind this project is to create a Deep Learning model that is able to detect whether a brain is healthy or not. If this model is accurate in detecting brain tumors, it can be a huge advantage in helping medical teams in detecting brain tumors despite the size or stages.

## **Model**
Model uses CNN method with Max pooling, Batch Normalization, Dropout techniques to prevent overfitting due to it's fairly complex characteristics. It then uses 2 fully connected layers as classifiers.

Training Hyperparameters:
* Learning Rate = 0.01
* batch_size = 128
* Weight_decay = 0.00001
* Dropout Probability= 0.4

Final model:
* Test Accuracy reached ~85%.
* Test Loss: 0.189, Test Error: 0.015,
* Train Loss: 0.230, Train Error: 0.068, Train Acc: 90.25%
* Val Loss: 0.300, Val Error: 0.026, Val Acc: 87.07%

## **Future Research**
Further considerations to be made in the future:
* A more complex model that uses techniques such as ResNet, Inception.
* More data points.
* Data augmentation techniques such as rotating and flipping.
