## Model API

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, request , render_template
import cv2
import speech_recognition as sr
from matplotlib import animation as animation, pyplot as plt, cm
from moviepy.editor import VideoFileClip
from pydub import AudioSegment 
from pydub.utils import make_chunks 
import librosa

new_model = tf.keras.models.load_model('C:/Users/kunal/Desktop/CAPSTONE_PROJECT/Demo/model_no_val_94.05.h5')
audio_model = tf.keras.models.load_model('C:/Users/kunal/Desktop/CAPSTONE_PROJECT/Demo/audio_model_98_85.h5')

def prepare_image(img_path):
    pred_array = []
    Output = {'Angry': 0, 'Disgust' : 0, 'Fear' : 0,'Happy' : 0,'Neutral' : 0, 'Sad' : 0, 'Surprise' : 0}
    neu = 0
    happy = 0
    sad = 0
    dis = 0
    fear = 0
    ang = 0
    sur = 0
    count = 0 
    pred = 0  
    vidcap = cv2.VideoCapture(img_path)
    total_frame = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    seconds = 3
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    multiplier = fps * seconds
    #ret,frame = vidcap.read()

    # Check if camera opened successfully
    if (vidcap.isOpened()== False):
        print("Error opening video stream or file")
    
    frame_counter = 1

    while frame_counter <= total_frame:     
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
        ret,frame = vidcap.read()
        path = "C:/Users/kunal/Desktop/CAPSTONE_PROJECT/Demo/static/frame/"
        if (ret):
            cv2.imwrite(path+"%d.jpg" % count,frame)
        else:
            break
        frame_counter += multiplier
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
                        pred_array.append("Angry")
                    elif pred == 1:
                        pred_array.append("Disgust")
                    elif pred == 2:
                        pred_array.append("Fear")
                    elif pred == 3 :
                        pred_array.append("Happy")
                    elif pred == 5:
                        pred_array.append("Sad")
                    elif pred == 6:
                        pred_array.append("Surprise")
                    else:
                        pred_array.append("Neutral")

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
    pred_array.append("Neutral")
    print('Output Video: ',Output)
    return (pred,Output,pred_array,count)

def prepare_audio(img_path,output_ext="wav"):
    pred_audio = []
    paths = []
    audio_text = []
    Output = {'Angry': 0, 'Disgust' : 0, 'Fear' : 0,'Happy' : 0,'Neutral' : 0, 'Sad' : 0, 'Surprise' : 0}
    neu = 0
    happy = 0
    sad = 0
    dis = 0
    fear = 0
    ang = 0
    sur = 0
    q = "C:/Users/kunal/Desktop/CAPSTONE_PROJECT/Demo/static/chunks/"
    filename, ext = os.path.splitext(img_path)
    clip = VideoFileClip(img_path)
    clip.audio.write_audiofile(f"{filename}.{output_ext}")
    r = sr.Recognizer()
   
    myaudio = AudioSegment.from_file(filename+".wav", "wav") 
    chunk_length_ms = 2727.27 # pydub calculates in millisec 
    chunks = make_chunks(myaudio,chunk_length_ms) #Make chunks of one sec 

    for i, chunk in enumerate(chunks): 
        chunk_name = "{0}.wav".format(i) 
        #print ("exporting", chunk_name) 
        chunk.export(q+chunk_name, format="wav") 
        with sr.AudioFile(q+chunk_name) as source:
            audio_listened = r.record(source)
            # try converting it to text
            try:
                text = r.recognize_google(audio_listened)
                text = f"{text.capitalize()} "
                audio_text.append(text)
            except:
                print("Error")
                
    p = 'C:/Users/kunal/Desktop/Audio Dataset/Audio Sample'
    for dirname, _, filenames in os.walk(q):
        for filename in filenames:
            paths.append(os.path.join(dirname, filename))
        if len(paths) == 2800:
            break
    
    df = pd.DataFrame()
    df['speech'] = paths

    def extract_mfcc(filename):
        y, sr = librosa.load(filename, duration=3, offset=0.5)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return mfcc
    
    X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))

    X = [x for x in X_mfcc]
    X = np.array(X)
    X = np.expand_dims(X, -1)

    prediction_raw = audio_model.predict(X)
    prediction = np.argmax(prediction_raw, axis=1)

    for i in prediction:
        if i == 0:
            ang = ang+1
            Output.update({'Angry':ang})
            pred_audio.append("Angry")
        elif i == 1:
            dis = dis+1
            Output.update({'Disgust':dis})
            pred_audio.append("Disgust")
        elif i == 2:
            fear = fear+1
            Output.update({'Fear':fear})
            pred_audio.append("Fear")
        elif i == 3 :
            happy = happy+1
            Output.update({'Happy':happy})
            pred_audio.append("Happy")
        elif i == 5:
            sad = sad+1
            Output.update({'Sad':sad})
            pred_audio.append("Sad")
        elif i == 6:
            sur = sur+1
            Output.update({'Surprise':sur})
            pred_audio.append("Surprise")
        else:
            neu = neu+1
            Output.update({'Neutral':neu})
            pred_audio.append("Neutral")

    print("Output Audio :",Output)
    print("Audio Text Array:", audio_text)
    return(prediction,Output,audio_text,pred_audio)
        
app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    img = request.files['my_image']

    img_path = "C:/Users/kunal/Desktop/CAPSTONE_PROJECT/Demo/static/" + img.filename	
    img.save(img_path)

    p,Output,pred_arr,count = prepare_image(img_path)
    x1 = list(Output.keys())
    y1 = list(Output.values())

    pred_audio,Output_audio,audio_text,pred_audio = prepare_audio(img_path)
    x2 = list(Output_audio.keys())
    y2 = list(Output_audio.values())

    return render_template("index.html", 
                            prediction = p, 
                            labels1 = x1, values1 = y1, 
                            max=count+5, path= img_path, 
                            pred_array=pred_arr,
                            count=count-1,
                            pred_audio=pred_audio, 
                            labels2 = x2, values2 = y2, 
                            audio_text = audio_text, 
                            pred_audio_array = pred_audio)

if __name__ == '__main__':
    app.run(debug=True)
