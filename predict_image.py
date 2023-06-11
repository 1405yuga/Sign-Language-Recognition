from PIL import Image
import numpy as np
from tensorflow import keras


imgg=Image.open('C:/Users/DELL/OneDrive/Pictures/Camera Roll/imtest.jpg')
imgg=imgg.resize((28,28),Image.BICUBIC)
img_grey = imgg.convert('L')
value = np.asarray(img_grey.getdata()).reshape((img_grey.size[1], img_grey.size[0]))
print(np.shape(value))
value=value/255
value=np.expand_dims(value,axis=0)
value=np.expand_dims(value,axis=3)
print(np.shape(value))

model=keras.models.load_model('D:\Sem 5\Mini project\Project\sign_lang_bestModel.h5')
print(model.predict(value))
result=10
arr=np.array(model.predict(value))

        

    
print(np.any(arr>=0.6))