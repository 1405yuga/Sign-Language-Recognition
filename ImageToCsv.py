from PIL import Image
import numpy as np
import os
import csv


def imageTocsv(imagename,label,datatype):
    
    
    img_file = Image.open(imagename)
    #img_file.show()
    
    # get original image parameters...
    width, height = img_file.size
    
    img_file=img_file.resize((28,28),Image.BICUBIC)
    # Make image Greyscale
    img_grey = img_file.convert('L')
    #img_grey.save('result.png')
    #img_grey.show()
    value = np.asarray(img_grey.getdata()).reshape((img_grey.size[1], img_grey.size[0]))
    value = value.flatten()
    value=np.append(value,label)
    print(value)
    
    print(np.size(value),"----------------------------------------")
    with open("{}_data.csv".format(datatype), 'a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(value)
        
#Traversing through Folder train_set
def collectANDfilterDataset(Foldername,dataType):        
    #Foldername='D:/Sem 5/Mini project/Project/train_set'
    for eachFolder in os.listdir(Foldername):
        #eachFolder is folder 0,1,...,9
        nm=os.path.join(Foldername,eachFolder)
        AllImages=os.listdir(nm)
        
        #Traversing through eachfolder containing images
        for eachImage in AllImages:
            #applying imageTocsv function to each image
            imageTocsv(os.path.join(nm,eachImage),eachFolder,dataType)