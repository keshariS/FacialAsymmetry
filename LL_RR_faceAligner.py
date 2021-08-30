
from imutils import face_utils
from imutils.face_utils import FaceAligner
import numpy as np
import imutils
import dlib
import cv2
from skimage.metrics import structural_similarity as ssim
#import os
import glob
#import time
from tkinter import *
import tkinter as tk
from tkinter import filedialog
from pandas import DataFrame
#import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# loading models for face detection and set defaults
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
camera = cv2.VideoCapture(0)
main_option=1
# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=256)

#Beginning GUI
gui = Tk(className=' LLRR_Facial_Similarity')
# set window size and position it at center of screen
windowWidth=800
windowHeight=400
positionRight = int(gui.winfo_screenwidth()/2 - windowWidth/2)
positionDown = int(gui.winfo_screenheight()/2 - windowHeight/2)
gui.geometry("{}x{}+{}+{}".format(windowWidth,windowHeight,positionRight,positionDown))
xx=gui.winfo_screenwidth()/2

w = Label(gui, text="\nWelcome! \n\nThis tool helps to analyze similarity of dynamic composite faces\n",font=("Helvetica", 15))
w.pack()
v = IntVar()# identifies which one is selected

Label(gui, text="Select one of the following ways of capturing a video:",justify = LEFT,padx = 20).pack()
Radiobutton(gui, text="Real-time Analysis via webcam",padx = 20, variable=v, value=1).pack(anchor=W)
Radiobutton(gui, text="Analysis of a pre-recorded video",padx = 20, variable=v, value=2).pack(anchor=W)

def helloCallBack():
    global camera
    global main_option
    if v.get()==1:
        gui.destroy()
        tempp=Tk(className=' Note')
        # set window size and position it at center of screen
        winWidth=400
        winHeight=200
        posRight = int(tempp.winfo_screenwidth()/2 - winWidth/2)
        posDown = int(tempp.winfo_screenheight()/2 - winHeight/2)
        tempp.geometry("{}x{}+{}+{}".format(winWidth,winHeight,posRight,posDown))
        Label(tempp,text="\nWebCam Callibration Complete\n",font=("Helvetica", 10)).pack()
        Label(tempp,text="Press the button below to begin Real-time streaming!",font=("Helvetica", 10)).pack()
        Label(tempp,text="(Press q to stop recording anytime you wish)\n",font=("Helvetica", 10)).pack()
        B1 = Button(tempp, text="START", command = tempp.destroy)
        B1.pack()
        tempp.mainloop()
        
    if v.get()==2:
        root = Tk(className=' Choose Video...')
        root.geometry("500x100+10+10")#width x heigth
        w1 = Label(root, text="\nBrowse your system for the Test Video...",font=("Helvetica", 15))
        w1.pack()
        root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("All files","*.*"),("jpeg files","*.jpg")))
        test_video_path = root.filename
        root.destroy()
        
        camera = cv2.VideoCapture(test_video_path)# from the pre recorded video in path
        main_option=2
        gui.destroy()
        
        tempp=Tk(className=' Note')
        # set window size and position it at center of screen
        winWidth=400
        winHeight=200
        posRight = int(tempp.winfo_screenwidth()/2 - winWidth/2)
        posDown = int(tempp.winfo_screenheight()/2 - winHeight/2)
        tempp.geometry("{}x{}+{}+{}".format(winWidth,winHeight,posRight,posDown))
        Label(tempp,text="\nPreliminary Callibration Complete\n",font=("Helvetica", 10)).pack()
        Label(tempp,text="Press the button below to begin video analysis!",font=("Helvetica", 10)).pack()
        Label(tempp,text="(Press q to stop anytime you wish)\n",font=("Helvetica", 10)).pack()
        B1 = Button(tempp, text="START", command = tempp.destroy)
        B1.pack()
        tempp.mainloop()


button = Button(gui, text='Confirm', width=25, command=helloCallBack)
button.pack()

gui.mainloop()

# starting video streaming
cv2.namedWindow('TestVideo')
cv2.namedWindow('Aligned')
cv2.namedWindow('LL RR composites')
cv2.moveWindow('TestVideo', int(xx-400),75)# width wise centerscreen
tlt = 25 # number of pixels of tilt allowance (allow if <tlt)
t_pass = []
frms=0
sim_list = []

while camera.isOpened():
    ret, frame = camera.read()# by default the webcam reads at around 30fps, can be changed by other codes
    if ret==False:
        break
    #reading the frame
    frame = imutils.resize(frame,width=800)
    if main_option==1:
        frame = cv2.flip(frame, 1)
    frameClone = frame.copy()
    frameClone = cv2.putText(frameClone, 'Press Q to stop',(500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
    t_pass.append(frms)
    sim_list.append(-0.1)
    frms = frms+1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):# press q to stop
        break
        
    ###-------------begin finding 68 facial landmarks using dlib

    ## this section checks for correct facial alignment
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image (dlib object for dlib shape prediction)
    rects = detector(gray_frame, 1)
    if len(rects)!=0:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray_frame, rects[0])
        shape = face_utils.shape_to_np(shape)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(frameClone, (x, y), 1, (0, 0, 255), -1)

        ### ADD CODE for checking alignment
        ylj = shape[0][1] # y coordinate of left jaw
        yrj = shape[16][1] # y coordinate of right jaw
        xtn = shape[27][0] # x coordinate of top of nose
        xbn = shape[30][0] # x coordinate of bottom of nose

        faceAligned = fa.align(frame, gray_frame, rects[0])
        cv2.imshow('Aligned',faceAligned)

        if abs(ylj-yrj)>=tlt or abs(xtn-xbn)>=tlt:
            cv2.imshow('TestVideo', frameClone)
            continue
        
        # convert dlib's rectangle to a OpenCV-style bounding box [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rects[0])
        cv2.rectangle(frameClone, (x, y), (x + w, y + h), (0, 255, 0), 1)
        
        ###-------------end finding 68 facial landmarks using dlib
        
        ### using CNN : (if face is well aligned)
        # grab the frame dimensions and convert it to a blob
        (h, w) = faceAligned.shape[:2]
        #blob = cv2.dnn.blobFromImage(cv2.resize(faceAligned, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
        # pass the blob through the network and obtain the detections and predictions
        #net.setInput(blob)
        #detections = net.forward()
        #if detections[0, 0, 0, 2] > 0.75: # 75% confidence of a face existing in the frame
        # compute the (x, y)-coordinates of the bounding box for the object
        #box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
        #(startX, startY, endX, endY) = box.astype("int")
        #(fX, fY, fW, fH) = (startX, startY, endX-startX, endY-startY)
        #cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),(255, 0, 0), 1)
        
        #crop_face = faceAligned.copy()#[startY:endY, startX:endX]
        crop_face = faceAligned[h//10:h*9//10, w//10:w*9//10]
        #----------------------LL RR--------------
        (hh,ww,dd) = crop_face.shape
        if ww%2==0:
            ww1=ww//2-1
        else:
            ww1=ww//2
        flipHorizontal = cv2.flip(crop_face, 1)
        img1 = crop_face[:,0:ww1]
        img2 = flipHorizontal[:,ww1+1:]
        LL = np.concatenate((img1, img2), axis=1)
        img1 = flipHorizontal[:,0:ww1]
        img2 = crop_face[:,ww1+1:]
        RR = np.concatenate((img1, img2), axis=1)
        llrr = np.concatenate((LL,RR),axis=0)
        cv2.imshow('LL RR composites',llrr)
        
        # calculate similarity index (0-1) (least - identical)
        sim_index = ssim(cv2.cvtColor(LL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(RR, cv2.COLOR_BGR2GRAY))
        sim_list[frms-1] = sim_index
        
        cv2.imshow('TestVideo', frameClone)
    
    else:
        cv2.imshow('TestVideo', frameClone)
        continue
        
camera.release()
cv2.destroyAllWindows()

t_passn =np.array(t_pass)
t_passn =100*t_passn/t_pass[-1]
sim_listn = 100*np.array(sim_list) # percentage

data = {'Time': t_passn,
         'Similarity_index': sim_listn
        }
df2 = DataFrame(data,columns=['Time','Similarity_index'])

res = Tk(className=' Final Results')
# set window size and position it at center of screen
#winWidth=900
#winHeight=550
#posRight = int(res.winfo_screenwidth()/2 - winWidth/2)
#posDown = int(res.winfo_screenheight()/2 - winHeight/2)
#res.geometry("{}x{}+{}+{}".format(winWidth,winHeight,posRight,posDown))

figure2 = plt.Figure(figsize=(8,6), dpi=100)
ax2 = figure2.add_subplot(111)
line2 = FigureCanvasTkAgg(figure2, res)# using toplevel for graph
line2.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
df2 = df2[['Time','Similarity_index']].groupby('Time').sum()
df2.plot(kind='line', legend=True, ax=ax2,fontsize=10)
ax2.set_title('Variation of Similarity index over captured frames')

res.mainloop()