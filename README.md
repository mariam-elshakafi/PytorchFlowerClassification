# PytorchFlowerClassification
A simple python program to classify 102 flower data using Pytorch.

The program prompts user to choose a json file (which has a mapping to flower names), 
then a flower picture, then attempts to classify it. The top 5 classes are shown in a bar chart.

![Prediction Results](../assets/ResultScreenshot.png?raw=true)

#### Training and Prediction

predict_out.py can be run directly using the attached checkpoint,
or train.py can be used to generate a checkpoint first.

This program has been tested using 2 architectures:
densenet, alexnet
Although the checkpoint attached is for densenet only.
