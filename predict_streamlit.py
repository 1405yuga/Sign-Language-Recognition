import numpy as np
from tensorflow import keras
import cv2
import streamlit as st
from PIL import Image
import timeit


model=keras.models.load_model('D:\Sem 5\Mini project\Project\sign_lang_bestModel.h5')

def predict(image):
    image=cv2.resize(image,(28,28))
    image=cv2.flip(image,1)
    image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image=image/255
    image=np.expand_dims(image,axis=0)
    image=np.expand_dims(image,axis=3)
   
    m=model.predict(image)
    result=10
    if(np.any(m>=0.9)):
        return(np.argmax(m))
    return result

def Start_cam(run,cam):
    FRAME_WINDOW=st.image([])
    while(run=="Start Camera"):
        _,frame=cam.read()
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame=cv2.flip(frame,1)
        
        #create region of intreset
        x1 = int(0.5*frame.shape[1])
        y1 = 10
        x2 = frame.shape[1]-10
        y2 = int(0.5*frame.shape[1])
        # Drawing the ROI
        # The increment/decrement by 1 is to compensate for the bounding box
        cv2.rectangle(frame, (x1-1, y1-10), (x2+1, y2+10), (255,255,0) ,4)
        # Extracting the ROI
        roi = frame[y1:y2, x1:x2]
        
        start=timeit.default_timer()
        #give image for prediction
        result=predict(roi)
        prediction = ('ZERO', 
                  'ONE', 
                  'TWO',
                  'THREE',
                  'FOUR',
                  'FIVE',
                  'SIX',
                  'SEVEN',
                  'EIGHT',
                  'NINE',
                  'EMPTY')
    
        # Displaying the predictions
        cv2.putText(frame, prediction[result], (x1+5,y2+6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)     
        FRAME_WINDOW.image(frame)
        end=timeit.default_timer()
        print(start-end)
        
    
    
def main():
    break_line='<hr>'
    main_title='<h1 style="font-family:Comic Sans MS; color:#1A2B92 ;text-align:center;">Sign Language Recognition</h1>'
    st.markdown(main_title,unsafe_allow_html=True)
    new_title = '<p style="font-family:sans-serif; color:Green; font-size: 34px;text-align:center;">A project to bridge a communication gap !!</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    note='<p style="font-family:sans-serif; color:red; font-size: 18px;">Note : Show hand gesture within yellow box</p>'
    img=Image.open("signs.jpg")
    st.image(img)
    
    
    st.markdown(break_line,unsafe_allow_html=True)
    run=st.radio("Get Started : ",["Aim","Start Camera","Help"])
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    cam=cv2.VideoCapture(0)
    if(run=="Aim"):
        tt='<h2 style="font-family:Comic Sans MS; color: #51077F ;text-align:center;">What we aim ?</h2>'
        st.markdown(tt,unsafe_allow_html=True)
        listt,im=st.columns(2)
        img1=Image.open("bg.jpg")
        instruction_list='<ol><li>To reduce the stress of deaf and dumb people while communicating and giving confidence while expressing themselves.</li> <li>The aim is to detect hand gestures of dumb and deaf people which will help normal people to understand their language .</li> </ol>'
        listt.markdown(instruction_list,unsafe_allow_html=True)
        im.image(img1)
    elif(run=="Start Camera"):
        st.markdown(note,unsafe_allow_html=True)
        Start_cam(run, cam)
    elif(run=="Help"):
        cam.release()
        tt='<h2 style="font-family:Comic Sans MS; color: #51077F ;text-align:center;">How to use ?</h2>'
        st.markdown(tt,unsafe_allow_html=True)
        im1,listt,im2=st.columns([1,2,1])
        im=Image.open("ask.png")
        instruction_list='<ol><li>Click on Start Camera button.</li> <li>Place hand within yellow box.</li> <li>Make the hand gestures that is to be predicted.</li><li>Prediction will be displayed.</li></ol>'
        listt.markdown(instruction_list,unsafe_allow_html=True)
        im1.image(im)
        im2.image(im)
    else:
        cam.release()
    
    
    
        
    
main()
