## Model API

import flask
import io
import string
from io import BytesIO
import time
import os
import numpy as np
import tensorflow as tf
import seaborn as sns
from PIL import Image
from flask import Flask, jsonify, request , render_template
import cv2
import base64
import json
from matplotlib.figure import Figure
from matplotlib import animation as animation, pyplot as plt, cm

new_model = tf.keras.models.load_model('C:/Users/kunal/Desktop/CAPSTONE PROJECT/Demo/mod_my_model_94p69.h5')

def prepare_image(img_path):
    Output = {'Angry': 0, 'Disgust' : 0, 'Fear' : 0,'Happy' : 0,'Neutral' : 0, 'Sad' : 0, 'Surprise' : 0}
    neu = 0
    happy = 0
    sad = 0
    dis = 0
    fear = 0
    ang = 0
    sur = 0
    count = 0
    vidcap = cv2.VideoCapture(img_path)
    vidcap.set(cv2.CAP_PROP_POS_MSEC,26.5*1000)
    ret,frame = vidcap.read()
    while (ret):     
        ret,frame = vidcap.read()
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        count = count+1
        print('Frame count:',count)
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(frame,1.1,4)
        
        for x,y,w,h in faces:
            
            #roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x,y),(x+w, y+h), (255,0,0), 2)
            facess = faceCascade.detectMultiScale(roi_color)
            if len(facess) == 0:
                print("Face not detected")
            else:
                for (ex,ey,ew,eh) in facess:
                    face_roi = roi_color[ey:ey+eh, ex:ex+ew]
                    
                    final_image = cv2.resize(face_roi,(224,224))
                    final_image = np.expand_dims(final_image,axis=0)
                    final_image = final_image/255.0
                    Predictions = new_model.predict(final_image)
                    pred = np.argmax(Predictions)
                    if pred == 0:
                        print("predictions : ",pred)
                        ang = ang+1
                        Output.update({'Angry':ang})
                    elif pred == 1:
                        print("predictions : ",pred)
                        dis = dis+1
                        Output.update({'Disgust':dis})
                    elif pred == 2:
                        print("predictions : ",pred)
                        fear = fear+1
                        Output.update({'Fear':fear})
                    elif pred == 3 :
                        print("predictions : ",pred)
                        happy = happy+1
                        Output.update({'Happy':happy})
                    elif pred == 5:
                        print("predictions : ",pred)
                        sad = sad+1
                        Output.update({'Sad':sad})
                    elif pred == 6:
                        print("predictions : ",pred)
                        sur = sur+1
                        Output.update({'Surprise':sur})
                    else:
                        print("predictions : ",pred)
                        neu = neu+1
                        Output.update({'Neutral':neu})
    print('Output Dictionary: ',Output)
    return (pred,Output)

def create_plot():
    Output = {'Angry': 20, 'Disgust': 27, 'Fear': 721, 'Happy': 244, 'Neutral': 379, 'Sad': 12, 'Surprise': 2}
    # Generate the figure **without using pyplot**.
    x = list(Output.keys())
    y = list(Output.values())
    fig = Figure()
    ax = fig.subplots()
    ax.bar(x,y)
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return (f"<img src='data:image/png;base64,{data}'/>")

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    img = request.files['my_image']

    img_path = "C:/Users/kunal/Desktop/CAPSTONE PROJECT/Demo/static/" + img.filename	
    img.save(img_path)

    p,Output = prepare_image(img_path)

    x = list(Output.keys())
    y = list(Output.values())

    return render_template("index.html", prediction = p, labels = x, values = y, max=600)

if __name__ == '__main__':
    app.run(debug=True)
