# Analysis-of-Deep-Learning-algorithms-on-COVID-19-Radiography-Database

This repository illustrates the implementation of the Research Paper titled "**Analysis-of-Deep-Learning-algorithms-on-COVID-19-Radiography-Database**" published in "**International Journal of Advanced Science And Technology (IJAST)**".

The below link can lead to the official site page:-
**Link** - http://sersc.org/journals/index.php/IJAST/article/view/20825

## Introduction

This repository aims to analyze the various deep learning algorithms on the radiography database that could be useful in diagnosing the COVID19 existence. We also discussed recently available datasets around the globe and the application of various DL architectures in the paper. LeNet5, CNN, Dense-Net121, DenseNet169, DenseNet201, ResNet50, VGG16, VGG19, MobileNetV2, NasNetMobile, NasNetLarge, InceptionV3, InceptionResnetv2 and Xception were presented with performance measures as a proof of concept. Further, we proposed a method to detect the COVID-19 presence based on the results of the above architectures. X-ray diagnosing can be used as an initial method during large population testing and can be made easily available at any remote place with a good internet connection.

## Dataset 

* You can find the dataset in the data folder being uploaded or can download from the below google drive link:-
* Link - "https://drive.google.com/drive/folders/1sJEAQydA_WAQfLV3mBq-pwZHz_5CRQPQ?usp=sharing"

## Steps to run the above-mentioned codes

1. Upload the data files into your drive link. Alternatively, you can train on your local system or can upload it on any drive. 
> NOTE:- Here I am using google colab GPU to train the models.
2. Open the jupyter notebook file in google drive and change the directory as per your path. 
3. Pre-trained Deep Learning architectures have been used to train. These DL architectures include:-
> **VGG16**
> /
> **VGG19**
> /
> **ResNet50**
> /
> **DenseNet121**
> /
> **DenseNet169**
> /
> **DenseNet201**
> /
> **MobileNetV2**
> /
> **NasNetMobile**
> /
> **NasNetLarge**
> /
> **InceptionV3**
> /
> **InceptionResNetV2**
> /
> **Xception**
4. Run each notebook file on google colab and model weights will be saved in your drive.
5. Call these trained weights while diagnosing COVID-19 with a new x-ray image as per your choice.
