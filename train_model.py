import pandas as pd
import numpy as np
from tensorflow import keras

training_data=pd.read_csv('D:/Sem 5/Mini project/Project/train_data.csv')
testing_data=pd.read_csv('D:/Sem 5/Mini project/Project/test_data.csv')

(rows,cols)=training_data.shape
sq=int((cols-1)**0.5)
(r,c)=testing_data.shape
s=int((c-1)**0.5)

#getting labels
yTrain_labels=training_data.iloc[:,cols-1:]
yTest_labels=testing_data.iloc[:,c-1:]


xTrain_images_vector=training_data.iloc[:,0:cols-1]
xTest_images_vector=testing_data.iloc[:,0:c-1]

#reshaping to image and appending for xTrain
xTrain_images=[]
for i in range(rows):
    #each image
    x=np.array(xTrain_images_vector.iloc[i])
    Image=x.reshape(sq,sq)
    xTrain_images.append(Image)
    
xTest_images=[]
for i in range(r):
    #each image
    x=np.array(xTest_images_vector.iloc[i])
    Image=x.reshape(s,s)
    xTest_images.append(Image)

xTrain_images=np.array(xTrain_images)  
xTrain_images=xTrain_images.reshape(len(xTrain_images),sq,sq,1) 

xTest_images=np.array(xTest_images)  
xTest_images=xTest_images.reshape(len(xTest_images),s,s,1) 


#Normalizing
xTrain_images=xTrain_images/255
xTest_images=xTest_images/255


#model training
model = keras.Sequential([
    #CNN layer
    keras.layers.Conv2D(filters=28,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)),
    keras.layers.MaxPooling2D((2,2)),
    
    keras.layers.Conv2D(filters=56,kernel_size=(3,3),activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    
    keras.layers.Conv2D(filters=112,kernel_size=(3,3),activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    
   
    #ANN layer
        keras.layers.Flatten(),
        keras.layers.Dense(56, activation='relu'),
        keras.layers.Dense(112, activation='relu'),
        keras.layers.Dense(10, activation='softmax')    
    ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(xTrain_images, yTrain_labels, epochs=25)

model.evaluate(xTest_images, yTest_labels)
modelName='sign_lang_bestModel.h5'
model.save(modelName)










