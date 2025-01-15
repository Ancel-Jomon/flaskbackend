from flask import Flask,jsonify
import numpy as np
import threading
from body_keypoint_track import MEDIAPIPE_POSE_KEYPOINTS
from plot3d import shared_data, printvals

app=Flask(__name__)

@app.route('/')
def home():
   kpts3d=shared_data.kpts3d
   dict={}
   i=0
   shared_data.data_ready.wait()
   with shared_data.lock:
       kpts3d=shared_data.kpts3d
   for s in MEDIAPIPE_POSE_KEYPOINTS:
       
        if(kpts3d is None):
           print("no")
        else:
            
            dict[s]=kpts3d[i].tolist()
            i+=1
   
   return jsonify(dict)


if __name__=="__main__":

    background_thread=threading.Thread(target=printvals)
    background_thread.daemon=True
    background_thread.start()
    app.run(debug=True)




