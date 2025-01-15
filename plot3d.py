import threading
import cv2
from body_keypoint_track import MEDIAPIPE_POSE_CONNECTIONS, BodyKeypointTrack, test
import numpy as np
import matplotlib.pyplot as plt


# test()


class SharedData:
    def __init__(self):
        self.kpts3d = None
        self.visib = None
        self.lock = threading.Lock()
        self.data_ready = threading.Event()

shared_data=SharedData()
def printvals():
    
    INPUT_FILE = 'C:\\Users\\ANCEL PUTHOOR\\Downloads\\dance.mp4'
    INPUT_IMAGE_SIZE = (640, 360)
    cap = cv2.VideoCapture(INPUT_FILE)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
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
        print(kpts3d)
        with shared_data.lock:
            shared_data.kpts3d=kpts3d
            shared_data.visib=visib
        shared_data.data_ready.set()
        break
        # kpts3ds.append((kpts3d, visib))

        frame_t +=1.0 / frame_rate
        frame_i += 1
    cap.release()
    cv2.destroyAllWindows()


def test2():

    plt.ion()  # Enable interactive mode
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')


    INPUT_FILE = 'C:\\Users\\ANCEL PUTHOOR\\Downloads\\dance.mp4'
    INPUT_IMAGE_SIZE = (640, 360)
    cap = cv2.VideoCapture(INPUT_FILE)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
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
        kpts3ds.append((kpts3d, visib))
        ax.clear()
        
        # Plot visible keypoints
        visible_points = kpts3d[visib == 1]
        ax.scatter(visible_points[:, 0], visible_points[:, 1], visible_points[:, 2], 
                  c='red', marker='o', s=50)
        
      
        for a, b in MEDIAPIPE_POSE_CONNECTIONS:
            if visib[a] == 1 and visib[b] == 1:
                ax.plot([kpts3d[a, 0], kpts3d[b, 0]],
                       [kpts3d[a, 1], kpts3d[b, 1]],
                       [kpts3d[a, 2], kpts3d[b, 2]], c='blue', linewidth=2)
        
        # Set consistent view and limits
        ax.view_init(elev=-90, azim=-90)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {frame_i}')
        
        # Equal aspect ratio
        ax.set_box_aspect([1,1,1])
        
       
        
        # Update display
        plt.draw()
        plt.pause(0.001) 
        frame_t +=1.0 / frame_rate
        frame_i += 1
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()  # Disable interactive mode
    plt.close()


# test2()