# from flask import Flask

# app=Flask(__name__)

# @app.route('/')
# def home():
#     return "hello"

# if __name__=="__main__":
#     app.run(debug=True)


import cv2
from body_keypoint_track import MEDIAPIPE_POSE_CONNECTIONS, BodyKeypointTrack, test
import numpy as np
import matplotlib.pyplot as plt

# test()


def test2():

    plt.ion()  # Enable interactive mode
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')


    INPUT_FILE = 'C:\\Users\\ANCEL PUTHOOR\\Downloads\\dance.mp4'
    INPUT_IMAGE_SIZE = (640, 360)
    cap = cv2.VideoCapture(INPUT_FILE)
    kpts3ds = []
    
    body_keypoint_track = BodyKeypointTrack(
        *INPUT_IMAGE_SIZE, 
        np.pi / 4, 
        track_hands=False, 
        smooth_range=0.3, 
        smooth_range_barycenter=1.0, 
        frame_rate=1.0 / 30.0
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
        ax.scatter(visible_points[:, 2], visible_points[:, 1], visible_points[:, 0], 
                  c='red', marker='o', s=50)
        
      
        for a, b in MEDIAPIPE_POSE_CONNECTIONS:
            if visib[a] == 1 and visib[b] == 1:
                ax.plot([kpts3d[a, 2], kpts3d[b, 2]],
                       [kpts3d[a, 1], kpts3d[b, 1]],
                       [kpts3d[a, 0], kpts3d[b, 0]], c='blue', linewidth=2)
        
        # Set consistent view and limits
        ax.view_init(elev=10, azim=45)
        
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
        frame_t += 1/30
        frame_i += 1
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()  # Disable interactive mode
    plt.close()


test2()