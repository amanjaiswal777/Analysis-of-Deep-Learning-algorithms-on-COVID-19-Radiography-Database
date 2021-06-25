# Analysis-of-Deep-Learning-algorithms-on-COVID-19-Radiography-Database

This repository illustrates the implementation of the Research Paper titled "**Analysis-of-Deep-Learning-algorithms-on-COVID-19-Radiography-Database**" published in "**International Journal of Advanced Science And Technology (IJAST)**".

The below link can lead to the official site page:-
**Link** - http://sersc.org/journals/index.php/IJAST/article/view/20825

## Introduction

This repository aims to analyze the various deep learning algorithms on the radiography database that could be useful in diagnosing the COVID19 existence. We also discussed recently available datasets around the globe and the application of various DL architectures in the paper. LeNet5, CNN, Dense-Net121, DenseNet169, DenseNet201, ResNet50, VGG16, VGG19, MobileNetV2, NasNetMobile, NasNetLarge, InceptionV3, InceptionResnetv2 and Xception were presented with performance measures as a proof of concept. Further, we proposed a method to detect the COVID-19 presence based on the results of the above architectures. X-ray diagnosing can be used as an initial method during large population testing and can be made easily available at any remote place with a good internet connection.

## Dataset 

* You can find sample dataset in the data folder being uploaded and for full dataset please download through below google drive link:-
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

## Alternative (Direct downloading trained weights files using google drive link)

1. VGG16_Covid_trained_weight = https://drive.google.com/file/d/1Yq2_2TbRG3Cm_A02hjF2gJOTV3-omObA/view?usp=sharing
2. VGG19_Covid_trained_weight = https://drive.google.com/file/d/1wZ2MweOKqPWVPi1BQiH3vQ2Oz2CmjT1i/view?usp=sharing
3. ResNet50_Covid_trained_weight = https://drive.google.com/file/d/1Oul6gRyWKtWFCy3uiPUz1EKZoKAjfujF/view?usp=sharing
4. MobileNetV2_Covid_trained_weight = https://drive.google.com/file/d/1Kh3aJg-Q96C24Q_iex1TPW1ZaD6iq82_/view?usp=sharing
5. NasNetMobile_Covid_trained_weight = https://drive.google.com/file/d/1ppRdfER4ig77tcR1KQ0U5z0NzEF_XDiZ/view?usp=sharing
6. NasNetLarge_Covid_trained_weight = https://drive.google.com/file/d/163FNh4Rrd20npfoNXruNZir1oX4rfzTn/view?usp=sharing
7. DenseNet121_Covid_trained_weight = https://drive.google.com/file/d/1wVf1aEH9C8Zxkw_qZtr-Icsj_vcLFlFw/view?usp=sharing
8. DenseNet169_Covid_trained_weight = https://drive.google.com/file/d/1s7q_O8s2NlXiRZyiQxwZQmjzQq9lwvJj/view?usp=sharing
9. DenseNet201_Covid_trained_weight = https://drive.google.com/file/d/1i3NdY6IXy7vn2n5z5u8tp4HbEqtdOiP6/view?usp=sharing
10. InceptionV3_Covid_trained_weight = https://drive.google.com/file/d/1fwijJ0KXTH01bj86ZkRaywAfBqS6jiaw/view?usp=sharing
11. InceptionResNetV2_Covid_trained_weight = https://drive.google.com/file/d/1--3Va7dOiZwN23zXKKAX3OpKphv61WqX/view?usp=sharing
12. Xception_Covid_trained_weight = https://drive.google.com/file/d/1uVr8rF6bpSUzRPBapPOaBDSokqJ_DcA8/view?usp=sharing

> **Direct use them for prediction**

## Conclusion and Future work

Overall, the results show that deep learning architectures have a significant impact on the COVID19 x-ray datasets. We also talked about new datasets that have recently become available around the world, as well as the use of various DL architectures. As a proof of concept, performance measures were presented to LeNet5, CNN, Dense-Net121, DenseNet169, DenseNet201, ResNet50, VGG16, VGG19, MobileNetV2, NasNetMobile, NasNetLarge, InceptionV3, InceptionResnetv2 and Xception. On the basis of the results of the previous architectures, we have proposed a method to detect the existence of COVID-19. X-ray diagnosis can be utilised as a first line of defence during large-scale population testing, and it can be made easily accessible from any remote location with a decent internet connection. More data, including but not limited to x-ray scans, could be added in future investigations. Furthermore, COVID-19 diagnosis employing sonography (lung ultrasound) in combination with radiography can be utilised to improve detection power, as ultrasound frequency analysis using acoustic models is sufficient for detecting COVID-19 existence.

### Feel free to drop question or doubt in repository issues section in **case of any issue** or reach out at **amanjaiswal5728@gmail.com**.




