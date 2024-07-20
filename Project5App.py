import tkinter as tk
from tkinter import filedialog
from tkinter import *

from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

import pandas as pd
from sklearn.cluster import KMeans

def DressColourDetect(file_path):
    global color_name

    net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    image_path = file_path
    image = cv2.imread(image_path)
    height, width, channels = image.shape

    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    cropped_image = image

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == "person":
                cropped_image = image[y:y+h, x:x+w]

    cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

    pixels = cropped_image_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[0]

    index = ["color", "color_name", "hex", "R", "G", "B"]
    df = pd.read_csv('colors.csv', names = index, header=None)
    [R,G,B] = dominant_color
    [R,G,B] = [round(R),round(G),round(B)]
    min = 99999
    color_name = None
    for i in range(len(df)):
        cc = abs(R-int(df.loc[i,"R"])) + abs(G-int(df.loc[i,"G"])) + abs(B-int(df.loc[i,"B"]))
        if(cc<=min):
            min = cc
            color_name = df.loc[i,"color_name"]
    print("Dominant color (RGB):", color_name)

def InitializeModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model

top = tk.Tk()
top.geometry('1200x800')
top.title('Emotion Detector Plus')
top.configure(background = '#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
AGE_LIST = ["Infant(0-6)", "Child(7-12)", "Teenager(13-17)", "Adult(18-29)", "Middle-Aged(30-59)", "Senior-Citizen(60-120+)"]
ETHNICITY_LIST =["Asian", "Black", "Indian", "Others", "White"]
GENDER_LIST = ["Female", "Male"]

Emomodel = InitializeModel("models/model_ed.json","weights/model_weights.h5")
Agemodel = InitializeModel("models/model_age1.json","weights/age_model_weights1.h5")
Ethmodel = InitializeModel("models/model_ethnicity.json","weights/ethnicity_model_weights.h5")
Genmodel = InitializeModel("models/model_gender1.json","weights/gender_model_weights1.h5")

def Detect(file_path):
    global label_packed

    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image,1.3,5)
    try:
        for (x,y,w,h) in faces:
            fc = gray_image[y:y+h,x:x+w]
            roi = cv2.resize(fc,(48,48))
            roi2 = cv2.resize(fc,(64,64))
            predicted_emotion = EMOTIONS_LIST[np.argmax(Emomodel.predict(roi[np.newaxis,:,:,np.newaxis]))]
            predicted_age = AGE_LIST[np.argmax(Agemodel.predict(roi2[np.newaxis,:,:,np.newaxis]))]
            predicted_ethnicity = ETHNICITY_LIST[np.argmax(Ethmodel.predict(roi2[np.newaxis,:,:,np.newaxis]))]
            predicted_gender = GENDER_LIST[np.argmax(Genmodel.predict(roi2[np.newaxis,:,:,np.newaxis]))]
            DressColourDetect(file_path)
        res = f"Emotion: {predicted_emotion}\nAge: {predicted_age}\nEthnicity: {predicted_ethnicity}\nGender: {predicted_gender}\nDress Color: {color_name}"
        print(res)
        if predicted_age==AGE_LIST[0] or predicted_age==AGE_LIST[1] or predicted_age==AGE_LIST[5]:
            res = "The provided image is of a Child or Senior Citizen. (Only ages 10-60 are allowed for detection)"
        label1.configure(foreground="#011638",text = res)
    except:
            label1.configure(foreground="#011638",text = "Unable to detect image")

def show_detect_button(file_path):
     detect_b = Button(top,text=">>Detect<<", command=lambda:Detect(file_path),padx=10,pady=5)
     detect_b.configure(background="#364156",foreground="white", font=('arial',10,'bold'))
     detect_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
         file_path = filedialog.askopenfilename()
         uploaded = Image.open(file_path)
         uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
         im = ImageTk.PhotoImage(uploaded)
         sign_image.configure(image=im)
         sign_image.image = im
         label1.configure(text='')
         show_detect_button(file_path)
    except:
         pass
    
upload = Button(top,text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156",foreground='white',font=('arial',20,'bold'))
upload.pack(side='bottom',pady=50)
sign_image.pack(side='bottom',expand='True')
label1.pack(side='bottom',expand='True')
heading = Label(top,text='Emotion Detector Plus', pady=20, font=('arial',25,'bold'))
heading.configure(background="#CDCDCD",foreground="#364156")
heading.pack()
top.mainloop()