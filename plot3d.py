import threading
import cv2
from body_keypoint_track import BodyKeypointTrack
import numpy as np






class SharedData:
    def __init__(self):
        self.kpts3d = None
        self.visib = None
        self.lock = threading.Lock()
        self.data_ready = threading.Event()
        self.posebones=None

shared_data=SharedData()
def printvals():
    
    INPUT_FILE = 'C:\\Users\\ANCEL PUTHOOR\\Downloads\\absolute.mp4'
    INPUT_IMAGE_SIZE = (360, 640)
    cap = cv2.VideoCapture(INPUT_FILE)
    frame_rate = 30
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    kpts3ds = []
    
    body_keypoint_track = BodyKeypointTrack(
       im_width=frame_width,
        im_height=frame_height,
        fov=np.pi / 3, 
        track_hands=False, 
        smooth_range=10 * (1 / frame_rate), 
        smooth_range_barycenter=30 * (1 / frame_rate), 
        frame_rate=frame_rate
    )
    frame_t = 0.0
    frame_i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), INPUT_IMAGE_SIZE)
        body_keypoint_track.track(frame, frame_t)
        
        kpts3d, visib = body_keypoint_track.get_smoothed_3d_keypoints(frame_t)
        kpts3d[:, 1] =-( kpts3d[:, 1])
        kpts3d[:, 2] =-( kpts3d[:, 2])
        with shared_data.lock:
            shared_data.kpts3d=kpts3d
            shared_data.visib=visib
        shared_data.data_ready.set()
        

        frame_t +=1.0 / frame_rate
        frame_i += 1
    cap.release()
    cv2.destroyAllWindows()





