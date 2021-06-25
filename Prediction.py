import tensorflow as tf 
import numpy as np
from keras.preprocessing import image
import os

final=[]
for filename in os.listdir("model/"):
    file = "model/" + filename
    classifierLoad = tf.keras.models.load_model("vgg16c.model")


    test_image = image.load_img('c3.png', target_size = (224,224))
    #test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifierLoad.predict(test_image)
    # print(result)
    if result[0][1] > result[0][0]:
        # print("NORMAL")
	    final.append(1)
    else:
        # print("COVID-19")
	    final.append(0)
print(final)
m,n =0,0
for i in range(len(final)):
	if final[i] == 0:
		m+=1
	else:
		n+=1
if m>n:
	print("COVID")
elif m<n:
	print("NORMAL")				
